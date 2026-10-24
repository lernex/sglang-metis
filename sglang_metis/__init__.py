# SPDX-License-Identifier: Apache-2.0
"""SGLang support for Lernex's Metis-1.6 models (architecture ``Metis16ForCausalLM``).

Installing this package registers, through SGLang's ``sglang.srt.plugins`` entry point:

* the ``metis16`` config and the ``Metis16ForCausalLM`` architecture,
* the hybrid Mamba-2 cache layout (one state or KV cache per layer per pass),
* the ``metis`` reasoning parser (``--reasoning-parser metis``),
* the ``metis`` tool-call parser (``--tool-call-parser metis``).
"""

__version__ = "0.1.0"

ARCHITECTURE = "Metis16ForCausalLM"


def register() -> None:
    """Entry point SGLang calls in every process before it parses arguments. Idempotent."""
    from sglang.srt.configs.linear_attn_model_registry import (
        LinearAttnModelSpec,
        get_linear_attn_spec_by_arch,
        register_linear_attn_model,
    )
    from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY

    from sglang_metis.config import Metis16Config

    # SGLang builds the config itself, so no remote code runs and the layer ids it sizes its
    # pools by are the per-pass ones. Transformers learns the type too: SGLang asks it first.
    _CONFIG_REGISTRY["metis16"] = Metis16Config
    from transformers import AutoConfig

    try:
        AutoConfig.register("metis16", Metis16Config)
    except ValueError:
        pass  # already registered in this process
    if get_linear_attn_spec_by_arch(ARCHITECTURE) is None:
        register_linear_attn_model(
            LinearAttnModelSpec(
                config_class=Metis16Config,
                backend_class_name="sglang.srt.layers.attention.hybrid_linear_attn_backend.Mamba2AttnBackend",
                arch_names=[ARCHITECTURE],
                uses_mamba_radix_cache=True,
                support_mamba_cache=True,
            )
        )

    from sglang.srt.models.registry import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.models:
        from sglang_metis.model import Metis16ForCausalLM

        ModelRegistry.models[ARCHITECTURE] = Metis16ForCausalLM

    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    from sglang_metis.parser import MetisDetector, MetisReasoningDetector

    FunctionCallParser.ToolCallParserEnum.setdefault("metis", MetisDetector)
    ReasoningParser.DetectorMap.setdefault("metis", MetisReasoningDetector)
