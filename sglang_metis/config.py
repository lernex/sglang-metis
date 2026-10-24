# SPDX-License-Identifier: Apache-2.0
"""The Metis-1.6 config as SGLang sees it.

Every token runs the ``num_physical_layers`` blocks ``num_passes`` times, and each (pass, layer)
pair keeps its own Mamba state or KV cache. To SGLang that is a stack of
``num_passes * num_physical_layers`` layers whose weights repeat: ``num_hidden_layers`` is that
virtual count, ``full_attention_layer_ids`` and ``mamba_layer_ids`` index it, and
``layer_id = pass * num_physical_layers + layer`` addresses one slot.
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class Metis16Config(PretrainedConfig):
    model_type = "metis16"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 2560,
        num_hidden_layers: int = 20,
        attention_layer_indices: list[int] | None = None,
        num_attention_heads: int = 40,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        attention_pass_lora_rank: int = 8,
        mamba_d_state: int = 128,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_head_dim: int = 64,
        mamba_n_groups: int = 8,
        mamba_chunk_size: int = 256,
        mamba_norm_eps: float = 1e-5,
        n_streams: int = 4,
        mhc_pass_embedding_dim: int = 64,
        mhc_sinkhorn_iterations: int = 8,
        latent_dim: int = 1024,
        n_routed_experts: int = 192,
        num_experts_per_tok: int = 4,
        min_routed_k: int = 1,
        max_routed_k: int = 8,
        expert_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        route_feature_dim: int = 256,
        memory_dim: int = 256,
        num_passes: int = 2,
        max_passes: int = 5,
        continuation_logit_limit: float = 4.0,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 131072,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = 1,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        num_physical_layers: int | None = None,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.attention_layer_indices = list(attention_layer_indices or [])
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_pass_lora_rank = attention_pass_lora_rank
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_head_dim = mamba_head_dim
        self.mamba_n_groups = mamba_n_groups
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_norm_eps = mamba_norm_eps
        self.n_streams = n_streams
        self.mhc_pass_embedding_dim = mhc_pass_embedding_dim
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations
        self.latent_dim = latent_dim
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.min_routed_k = min_routed_k
        self.max_routed_k = max_routed_k
        self.expert_intermediate_size = expert_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.route_feature_dim = route_feature_dim
        self.memory_dim = memory_dim
        self.num_passes = num_passes
        self.max_passes = max_passes
        self.continuation_logit_limit = continuation_logit_limit
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        # The checkpoint's num_hidden_layers counts physical blocks; a saved SGLang config has
        # both, so a round trip keeps the physical count.
        self.num_physical_layers = num_physical_layers or num_hidden_layers
        self.num_hidden_layers = self.num_passes * self.num_physical_layers
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def to_dict(self):
        out = super().to_dict()
        out["num_hidden_layers"] = self.num_physical_layers
        out.pop("num_physical_layers", None)
        return out

    @property
    def mamba_intermediate_size(self) -> int:
        return self.hidden_size * self.mamba_expand

    @property
    def mamba_num_heads(self) -> int:
        return self.mamba_intermediate_size // self.mamba_head_dim

    def slot(self, pass_index: int, layer_index: int) -> int:
        return pass_index * self.num_physical_layers + layer_index

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            self.slot(p, i)
            for p in range(self.num_passes)
            for i in range(self.num_physical_layers)
            if i in self.attention_layer_indices
        ]

    @property
    def mamba_layer_ids(self) -> list[int]:
        return [
            self.slot(p, i)
            for p in range(self.num_passes)
            for i in range(self.num_physical_layers)
            if i not in self.attention_layer_indices
        ]

    @property
    def mamba2_cache_params(self):
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateShape,
            mamba2_state_dtype,
        )
        from sglang.srt.runtime_context import get_parallel

        shape = Mamba2StateShape.create(
            tp_world_size=get_parallel().attn_tp_size,
            intermediate_size=self.mamba_intermediate_size,
            n_groups=self.mamba_n_groups,
            num_heads=self.mamba_num_heads,
            head_dim=self.mamba_head_dim,
            state_size=self.mamba_d_state,
            conv_kernel=self.mamba_d_conv,
        )
        return Mamba2CacheParams(
            shape=shape, layers=self.mamba_layer_ids, dtype=mamba2_state_dtype(self)
        )
