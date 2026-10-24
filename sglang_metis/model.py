# SPDX-License-Identifier: Apache-2.0
"""Metis-1.6 (Logos and Praxis) for SGLang.

Semantics follow the Hugging Face port (``modeling_metis16.py``), the reference: every token
runs the stack of blocks ``num_passes`` times with shared weights, carries four residual streams
joined by pass-conditioned Sinkhorn mixes, runs a Mamba-2 or position-free GQA mixer and then a
latent mixture of experts, and keeps a per-token depth memory written after every attention
layer and at the end of every pass.

Cache design
------------
Only the mixers look across positions. Each (pass, layer) pair is its own SGLang layer id,
``pass * num_physical_layers + layer`` (see ``config.Metis16Config``), so the hybrid backend
gives every pass of every Mamba layer its own conv and SSM state and every pass of every
attention layer its own KV cache, while the weights exist once:

* a Mamba block owns one ``MambaMixer2`` and calls the Mamba-2 backend with the slot's id;
* an attention block owns the projections and one weightless ``RadixAttention`` per pass.

Radix prefix caching, chunked prefill and CUDA graphs therefore work as for any hybrid model.

Numerics
--------
FP32 where the reference keeps it: the routers and the balance bias, ``A``/``D``/``dt_bias``,
the SSM state, the Sinkhorn projection, the depth-memory attention and gates, the depth head and
every RMSNorm. Online quantization leaves the Mamba output projection in the model dtype: it is
2% of the weights and most of a quantized model's error.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, set_weight_attrs
from sglang.srt.utils.custom_op import register_custom_op

from sglang_metis.config import Metis16Config

# Checkpoint tensors of the adaptive-depth and adaptive-width heads. The released weights were
# trained under one fixed policy (two passes, four experts) and the reference never reads them.
UNUSED_CHECKPOINT_SUFFIXES = (".k_router.weight", ".k_router.bias", ".k_budget.multiplier")
UNUSED_CHECKPOINT_NAMES = ("depth_budget.multiplier",)


# ---------------------------------------------------------------------------- numerics


def _rms_norm(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in FP32 with the weight applied before the cast back, as the reference does."""
    wide = values.float()
    wide = wide * torch.rsqrt(wide.square().mean(dim=-1, keepdim=True) + eps)
    return (wide * weight.float()).to(values.dtype)


def _sinkhorn(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    log_matrix = logits.float()
    for _ in range(iterations):
        log_matrix = log_matrix - torch.logsumexp(log_matrix, dim=-1, keepdim=True)
        log_matrix = log_matrix - torch.logsumexp(log_matrix, dim=-2, keepdim=True)
    return log_matrix.exp()


def _fp32_linear(module: nn.Linear, values: torch.Tensor) -> torch.Tensor:
    bias = module.bias.float() if module.bias is not None else None
    return F.linear(values.float(), module.weight.float(), bias)


def metis_route(logits, selection_bias, top_k: int, max_k: int):
    """Expert choice as the reference makes it.

    The balance bias moves which experts are chosen, never how they are weighted: pick
    ``topk(logits + bias, max_k)[:top_k]``, then softmax over the chosen unbiased logits.
    """
    logits = logits.float()
    chosen = torch.topk(logits + selection_bias.float(), k=max_k, dim=-1).indices[..., :top_k]
    weights = torch.softmax(logits.gather(-1, chosen), dim=-1)
    return chosen, weights


# ---------------------------------------------------------------------------- modules


class MetisRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return _rms_norm(values, self.weight, self.eps)


class MetisHyperConnection(nn.Module):
    """Four-stream manifold hyper-connection.

    The 4x4 doubly stochastic mix and both weight vectors depend only on the pass, never on the
    token, so they are computed once after the weights load (``finalize``) and kept as buffers.
    """

    def __init__(self, config: Metis16Config) -> None:
        super().__init__()
        n = config.n_streams
        self.n_streams = n
        self.iterations = config.mhc_sinkhorn_iterations
        self.norm = MetisRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.read_logits = nn.Parameter(torch.zeros(n))
        self.mix_logits = nn.Parameter(torch.zeros(n, n))
        self.write_logits = nn.Parameter(torch.zeros(n))
        self.controller = nn.Linear(config.mhc_pass_embedding_dim, n * n + 2 * n, bias=True)
        passes = config.num_passes
        self.register_buffer("mix", torch.zeros(passes, n, n), persistent=False)
        self.register_buffer("read_weights", torch.zeros(passes, n), persistent=False)
        self.register_buffer("write_weights", torch.zeros(passes, n), persistent=False)

    @torch.no_grad()
    def finalize(self, pass_embeddings: torch.Tensor) -> None:
        n = self.n_streams
        for p in range(self.mix.shape[0]):
            # Same op order and dtypes as the reference: the controller runs in the weight dtype.
            delta = self.controller(pass_embeddings[p]).float()
            matrix_delta, read_delta, write_delta = torch.split(delta, (n * n, n, n), dim=-1)
            matrix = _sinkhorn(self.mix_logits.float() + matrix_delta.view(n, n), self.iterations)
            self.mix[p].copy_(matrix)
            self.read_weights[p].copy_(torch.softmax(self.read_logits.float() + read_delta, -1))
            self.write_weights[p].copy_(torch.softmax(self.write_logits.float() + write_delta, -1))

    def read(self, streams: torch.Tensor, pass_index: int):
        mixed = torch.einsum("oi,nid->nod", self.mix[pass_index].to(streams.dtype), streams)
        source = torch.einsum("i,nid->nd", self.read_weights[pass_index].to(streams.dtype), streams)
        return self.norm(source), mixed

    def write(self, mixed: torch.Tensor, update: torch.Tensor, pass_index: int) -> torch.Tensor:
        weights = self.write_weights[pass_index].to(update.dtype)
        return mixed + torch.einsum("i,nd->nid", weights, update)


class _MambaSlot(nn.Module):
    """One pass of a Mamba block as SGLang's layer bookkeeping sees it: a layer id, no weights.

    SGLang lists one module per layer id (``attention_layers[layer_id]``) for looped models and
    finds the Mamba call through it when a CUDA graph breaks around the scan; attention passes
    are ``RadixAttention`` objects, Mamba passes these.
    """

    def __init__(self, layer_id: int, owner: "MetisMamba") -> None:
        super().__init__()
        self.layer_id = layer_id
        object.__setattr__(self, "_owner", owner)  # not a submodule: the block owns the mixer

    def _forward_mamba(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        return get_attn_backend().linear_attn_backend.forward(
            mixer=self._owner.mamba,
            layer_id=self.layer_id,
            hidden_states=hidden_states,
            output=None,
            forward_batch=forward_batch,
            use_triton_causal_conv=True,
        )


@register_custom_op(mutates_args=["output"])
@register_split_op()
def metis_mamba2_with_output(hidden_states: torch.Tensor, output: torch.Tensor, layer_id: int) -> None:
    """The Mamba scan as a graph break: it reads this step's metadata, so it runs eagerly."""
    context = get_tc_piecewise_forward_context()
    metadata = get_attn_backend().linear_attn_backend.forward_metadata
    actual = metadata.num_prefill_tokens + (
        metadata.num_decodes * metadata.draft_token_num if metadata.is_target_verify else metadata.num_decodes
    )
    result = context.attention_layers[layer_id]._forward_mamba(hidden_states[:actual], context.forward_batch)
    output[:actual].view(result.shape).copy_(result)
    if output.shape[0] != actual:
        output[actual:].zero_()


breakable_metis_mamba2_with_output = eager_on_graph(True)(metis_mamba2_with_output)


class MetisMamba(nn.Module):
    """A Mamba-2 block: one ``MambaMixer2``, one Mamba state per pass (selected by layer id)."""

    def __init__(self, config: Metis16Config, layer_index: int, *, quant_config, prefix) -> None:
        super().__init__()
        self.slots = [config.slot(p, layer_index) for p in range(config.num_passes)]
        self.attn = nn.ModuleList(_MambaSlot(slot, self) for slot in self.slots)
        self.mamba = MambaMixer2(
            cache_params=config.mamba2_cache_params,
            hidden_size=config.hidden_size,
            use_conv_bias=True,
            use_bias=False,
            n_groups=config.mamba_n_groups,
            rms_norm_eps=config.mamba_norm_eps,
            activation="silu",
            quant_config=quant_config,
            prefix=f"{prefix}.mamba",
        )
        mixer = self.mamba
        # The checkpoint and the reference keep D and dt_bias in FP32; MambaMixer2 makes them in
        # the model dtype. Same shard loaders, wider storage.
        for name in ("D", "dt_bias"):
            old = getattr(mixer, name)
            loader = old.weight_loader
            new = nn.Parameter(torch.ones(old.shape, dtype=torch.float32))
            setattr(mixer, name, new)
            set_weight_attrs(new, {"weight_loader": loader})
        if quant_config is not None:
            out = mixer.out_proj
            mixer.out_proj = RowParallelLinear(
                mixer.intermediate_size,
                config.hidden_size,
                bias=False,
                input_is_parallel=True,
                quant_config=None,
                reduce_results=out.reduce_results,
                prefix=f"{prefix}.mamba.out_proj",
            )

    def forward(self, hidden_states, pass_index: int, forward_batch: ForwardBatch):
        slot = self.slots[pass_index]
        if is_in_breakable_cuda_graph() or is_in_tc_piecewise_cuda_graph():
            output = torch.empty_like(hidden_states)
            op = breakable_metis_mamba2_with_output if is_in_breakable_cuda_graph() else metis_mamba2_with_output
            op(hidden_states, output, slot)
            return output
        return self.attn[pass_index]._forward_mamba(hidden_states, forward_batch)


class MetisAttention(nn.Module):
    """Position-free GQA with a per-pass low-rank adapter on QKV, one KV cache per pass."""

    def __init__(self, config: Metis16Config, layer_index: int, *, quant_config, prefix) -> None:
        super().__init__()
        tp = get_parallel().attn_tp_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = max(1, config.num_key_value_heads // tp)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )
        self.out = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out",
        )
        # The adapter stays in the model dtype; it is rank 8 and costs nothing.
        self.pass_lora_down = ReplicatedLinear(
            config.hidden_size, config.attention_pass_lora_rank, bias=False,
            prefix=f"{prefix}.pass_lora_down",
        )
        self.pass_lora_up = QKVParallelLinear(
            config.attention_pass_lora_rank,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            prefix=f"{prefix}.pass_lora_up",
        )
        self.pass_gates = nn.Parameter(torch.zeros(config.max_passes))
        self.register_buffer("gate_values", torch.zeros(config.max_passes), persistent=False)
        self.attn = nn.ModuleList(
            RadixAttention(
                self.num_heads,
                self.head_dim,
                self.head_dim**-0.5,
                num_kv_heads=self.num_kv_heads,
                layer_id=config.slot(p, layer_index),
                quant_config=quant_config,
                prefix=add_prefix(f"attn_pass{p}", prefix),
            )
            for p in range(config.num_passes)
        )

    @torch.no_grad()
    def finalize(self) -> None:
        self.gate_values.copy_(torch.tanh(self.pass_gates))

    def forward(self, hidden_states, pass_index: int, forward_batch: ForwardBatch):
        adapter, _ = self.pass_lora_up(self.pass_lora_down(hidden_states)[0])
        gate = self.gate_values[pass_index].to(hidden_states.dtype)
        qkv, _ = self.qkv(hidden_states)
        qkv = qkv + gate * adapter
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        out, _ = self.out(self.attn[pass_index](q, k, v, forward_batch))
        return out


class MetisSharedExpert(nn.Module):
    def __init__(self, latent: int, intermediate: int, prefix: str) -> None:
        super().__init__()
        self.gate_up = ReplicatedLinear(latent, 2 * intermediate, bias=False, prefix=f"{prefix}.gate_up")
        self.down = ReplicatedLinear(intermediate, latent, bias=False, prefix=f"{prefix}.down")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(hidden)[0].chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)[0]


class MetisMoE(nn.Module):
    """Latent MoE: route on (latent, routing features), top-k of the fused experts plus a shared one."""

    def __init__(self, config: Metis16Config, layer_index: int, *, quant_config, prefix) -> None:
        super().__init__()
        latent, route = config.latent_dim, config.route_feature_dim
        self.top_k = config.num_experts_per_tok
        self.max_k = config.max_routed_k
        self.latent_down = ReplicatedLinear(config.hidden_size, latent, bias=False, prefix=f"{prefix}.latent_down")
        self.latent_up = ReplicatedLinear(latent, config.hidden_size, bias=False, prefix=f"{prefix}.latent_up")
        self.expert_router = nn.Linear(latent + route, config.n_routed_experts, bias=True, dtype=torch.float32)
        self.selection_bias = nn.Parameter(
            torch.zeros(config.n_routed_experts, dtype=torch.float32), requires_grad=False
        )
        self.expert_embeddings = nn.Parameter(torch.empty(config.n_routed_experts, route))
        self.shared_expert = MetisSharedExpert(
            latent, config.shared_expert_intermediate_size, f"{prefix}.shared_expert"
        )
        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=self.top_k,
            hidden_size=latent,
            intermediate_size=config.expert_intermediate_size,
            layer_id=layer_index,
            reduce_results=True,
            # The latent also feeds the shared expert, so the routed kernel must not overwrite it.
            inplace=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor, route_features: torch.Tensor):
        latent = self.latent_down(hidden_states)[0]
        route_input = torch.cat((latent, route_features.to(latent.dtype)), dim=-1)
        logits = _fp32_linear(self.expert_router, route_input)
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1.0e-9).log()).sum(dim=-1)
        confidence = probabilities.amax(dim=-1)
        chosen, weights = metis_route(logits, self.selection_bias, self.top_k, self.max_k)
        routed = self.experts(
            latent,
            StandardTopKOutput(topk_weights=weights, topk_ids=chosen.to(torch.int32), router_logits=logits),
        )
        output = self.latent_up(routed + self.shared_expert(latent))[0]
        embeddings = self.expert_embeddings
        summary = (weights.to(embeddings.dtype).unsqueeze(-1) * embeddings[chosen]).sum(dim=-2)
        return output, (summary, entropy, confidence)


class MetisDepthMemory(nn.Module):
    """The per-token memory of its own trajectory through the passes.

    Keys and values are projected once per entry, when it is written, instead of re-projecting
    the whole bank on every read; the math is the same.
    """

    def __init__(self, config: Metis16Config) -> None:
        super().__init__()
        d, m, r = config.hidden_size, config.memory_dim, config.route_feature_dim
        self.hidden_size, self.memory_dim = d, m
        self.max_k_fraction = float(config.num_experts_per_tok) / float(config.max_routed_k)
        self.entropy_normalizer = math.log(config.n_routed_experts) if config.n_routed_experts > 1 else 1.0
        self.state_write = nn.Linear(d, m, bias=True)
        self.metadata_write = nn.Linear(r + 4, m, bias=True)
        self.pass_embeddings = nn.Parameter(torch.empty(config.max_passes, m))
        self.anchor_embeddings = nn.Parameter(torch.empty(len(config.attention_layer_indices) + 1, m))
        self.query = nn.Linear(d, m, bias=True)
        self.key = nn.Linear(m, m, bias=True)
        self.value = nn.Linear(m, m, bias=True)
        self.output = nn.Linear(m, d, bias=True)
        self.route_projection = nn.Linear(3 * d + r, r, bias=True)
        self.stream_gate = nn.Parameter(torch.zeros(config.n_streams, d))
        self.stream_gate_bias = nn.Parameter(torch.zeros(config.n_streams))

    def features(self, state, retrieved, difference, history):
        return torch.tanh(self.route_projection(torch.cat((state, retrieved, difference, history), dim=-1)))

    def retrieve(self, keys: list[torch.Tensor], values: list[torch.Tensor], streams):
        if not keys:
            return streams, streams.new_zeros(streams.shape[0], self.hidden_size)
        key = torch.stack(keys, dim=1)  # [N, m, M]
        value = torch.stack(values, dim=1)
        query = self.query(streams)  # [N, S, M]
        scores = torch.einsum("nsh,nmh->nsm", query.float(), key.float())
        weights = torch.softmax(scores / math.sqrt(self.memory_dim), dim=-1)
        retrieved = torch.einsum("nsm,nmh->nsh", weights, value.float()).to(value.dtype)
        projected = self.output(retrieved)
        gate_logits = (streams.float() * self.stream_gate.float()).sum(dim=-1) / math.sqrt(self.hidden_size)
        gates = torch.sigmoid(gate_logits + self.stream_gate_bias.float())
        fused = streams + gates[..., None].to(streams.dtype) * projected
        return fused, projected.mean(dim=1)

    def read(self, keys, values, streams, difference, history):
        streams, summary = self.retrieve(keys, values, streams)
        return streams, self.features(streams.mean(dim=1), summary, difference, history)

    def write(self, keys, values, streams, route, continuation, pass_index, anchor_index):
        summary, entropy, confidence = route
        state = streams.mean(dim=1)
        mean_k = torch.full_like(entropy, self.max_k_fraction)
        metadata = torch.cat(
            (
                summary.float(),
                mean_k.unsqueeze(-1),
                (entropy / self.entropy_normalizer).unsqueeze(-1),
                confidence.unsqueeze(-1),
                continuation.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        entry = (
            self.state_write(state)
            + self.metadata_write(metadata.to(state.dtype))
            + self.pass_embeddings[pass_index]
            + self.anchor_embeddings[anchor_index]
        )
        keys.append(self.key(entry))
        values.append(self.value(entry))


class MetisContinuation(nn.Module):
    """The depth head. Under the fixed pass count its output is only a memory feature."""

    def __init__(self, config: Metis16Config) -> None:
        super().__init__()
        d, r = config.hidden_size, config.route_feature_dim
        self.limit = config.continuation_logit_limit
        self.hidden = nn.Linear(3 * d + r, r, bias=True, dtype=torch.float32)
        self.output = nn.Linear(r, 1, bias=True, dtype=torch.float32)

    def forward(self, state, retrieved, difference, route_features):
        features = torch.cat((state, retrieved, difference, route_features), dim=-1)
        hidden = F.silu(_fp32_linear(self.hidden, features))
        logits = _fp32_linear(self.output, hidden).squeeze(-1)
        scale = state.float().square().mean(dim=-1).sqrt().clamp_min(1.0e-6)
        innovation = difference.float().square().mean(dim=-1).sqrt()
        logits = logits + torch.tanh(innovation / scale) - 0.5
        return torch.sigmoid(logits.clamp(-self.limit, self.limit))


class MetisBlock(nn.Module):
    def __init__(self, config: Metis16Config, layer_index: int, *, quant_config, prefix) -> None:
        super().__init__()
        self.is_attention = layer_index in config.attention_layer_indices
        mixer = MetisAttention if self.is_attention else MetisMamba
        self.mixer = mixer(config, layer_index, quant_config=quant_config, prefix=f"{prefix}.mixer")
        self.mixer_connection = MetisHyperConnection(config)
        self.moe = MetisMoE(config, layer_index, quant_config=quant_config, prefix=f"{prefix}.moe")
        self.moe_connection = MetisHyperConnection(config)

    def forward(self, streams, route_features, pass_index: int, forward_batch: ForwardBatch):
        mixer_input, mixed = self.mixer_connection.read(streams, pass_index)
        update = self.mixer(mixer_input, pass_index, forward_batch)
        streams = self.mixer_connection.write(mixed, update, pass_index)
        moe_input, mixed = self.moe_connection.read(streams, pass_index)
        moe_output, route = self.moe(moe_input, route_features)
        return self.moe_connection.write(mixed, moe_output, pass_index), route


# ---------------------------------------------------------------------------- model


class Metis16Model(nn.Module):
    def __init__(self, config: Metis16Config, quant_config: QuantizationConfig | None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.num_passes = config.num_passes
        self.embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size, prefix=f"{prefix}.embedding")
        self.stream_embeddings = nn.Parameter(torch.empty(config.n_streams, config.hidden_size))
        self.pass_embeddings = nn.Parameter(torch.empty(config.max_passes, config.mhc_pass_embedding_dim))
        self.depth_memory = MetisDepthMemory(config)
        self.layers = nn.ModuleList(
            MetisBlock(config, i, quant_config=quant_config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_physical_layers)
        )
        self.continuation = MetisContinuation(config)
        self.final_norm = MetisRMSNorm(config.hidden_size, config.rms_norm_eps)

    @torch.no_grad()
    def finalize(self) -> None:
        for layer in self.layers:
            layer.mixer_connection.finalize(self.pass_embeddings)
            layer.moe_connection.finalize(self.pass_embeddings)
            if layer.is_attention:
                layer.mixer.finalize()

    def forward(self, input_ids, positions, forward_batch: ForwardBatch, input_embeds=None):
        embeds = input_embeds if input_embeds is not None else self.embedding(input_ids)
        memory = self.depth_memory
        streams = embeds.unsqueeze(1) + self.stream_embeddings.to(embeds.dtype)  # [N, S, D]
        previous_state = streams.mean(dim=1)
        route_history = embeds.new_zeros(embeds.shape[0], self.config.route_feature_dim)
        continuation = embeds.new_zeros(embeds.shape[0])
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        # Inside a pass the reference reads memory with difference = mean(streams) - the state the
        # last pass ended on, taken before any layer runs: always exactly zero.
        no_difference = torch.zeros_like(previous_state)

        for pass_index in range(self.num_passes):
            anchor = 0
            route = None
            for block in self.layers:
                streams, features = memory.read(keys, values, streams, no_difference, route_history)
                streams, route = block(streams, features, pass_index, forward_batch)
                route_history = 0.5 * route_history + 0.5 * route[0]
                if block.is_attention:
                    memory.write(keys, values, streams, route, continuation, pass_index, anchor)
                    anchor += 1
            streams, summary = memory.retrieve(keys, values, streams)
            if pass_index + 1 == self.num_passes:
                break  # the last pass's depth head and memory entry are never read again
            current_state = streams.mean(dim=1)
            difference = current_state - previous_state
            features = memory.features(current_state, summary, difference, route_history)
            continuation = self.continuation(current_state, summary, difference, features).to(streams.dtype)
            memory.write(keys, values, streams, route, continuation, pass_index, anchor)
            previous_state = current_state

        return self.final_norm(streams.mean(dim=1))


class Metis16ForCausalLM(nn.Module):
    """SGLang entry point for ``Metis16ForCausalLM`` checkpoints (Metis-1.6 Logos and Praxis)."""

    fall_back_to_pt_during_load = False

    def __init__(self, config: Metis16Config, quant_config: QuantizationConfig | None = None, prefix: str = "") -> None:
        super().__init__()
        if get_parallel().pp_size > 1:
            raise ValueError(
                "Metis-1.6 runs every layer twice per token, so it cannot be split into "
                "pipeline stages; use tensor parallelism instead."
            )
        self.config = config
        self.quant_config = quant_config
        self.model = Metis16Model(config, quant_config, prefix=add_prefix("model", prefix))
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head = self.model.embedding
        else:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=add_prefix("lm_head", prefix))
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embedding

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch: ForwardBatch, input_embeds=None, **_):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    # ------------------------------------------------------------------ weights

    def map_checkpoint_name(self, name: str) -> str | None:
        """Where a checkpoint tensor goes: ``None`` to skip it, else this model's parameter name."""
        if name in UNUSED_CHECKPOINT_NAMES or name.endswith(UNUSED_CHECKPOINT_SUFFIXES):
            return None
        if name == "lm_head.weight":
            return None if getattr(self.config, "tie_word_embeddings", True) else name
        name = "model." + name
        if ".mixer.impl.mixer." in name:
            name = name.replace(".mixer.impl.mixer.", ".mixer.mamba.")
            if name.endswith(".A_log"):
                name = name[: -len("A_log")] + "A"
        elif ".moe.local_experts.gate_up.weight" in name:
            name = name.replace(".local_experts.gate_up.weight", ".experts.w13_weight")
        elif ".moe.local_experts.down.weight" in name:
            name = name.replace(".local_experts.down.weight", ".experts.w2_weight")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for checkpoint_name, tensor in weights:
            name = self.map_checkpoint_name(checkpoint_name)
            if name is None:
                continue
            if name not in params:
                raise KeyError(f"{checkpoint_name} maps to {name}, which this model does not have")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            # Stacked experts (and, in an FP8 checkpoint, their per-channel scales) go in one
            # expert and one half at a time.
            if name.endswith(("w13_weight", "w13_weight_scale")):
                half = tensor.shape[1] // 2
                for expert in range(tensor.shape[0]):
                    loader(param, tensor[expert, :half], name, shard_id="w1", expert_id=expert)
                    loader(param, tensor[expert, half:], name, shard_id="w3", expert_id=expert)
            elif name.endswith(("w2_weight", "w2_weight_scale")):
                for expert in range(tensor.shape[0]):
                    loader(param, tensor[expert], name, shard_id="w2", expert_id=expert)
            else:
                loader(param, tensor)
            loaded.add(name)
        self.model.finalize()
        return loaded


EntryClass = Metis16ForCausalLM
