# SPDX-License-Identifier: Apache-2.0
"""Every tensor in a release checkpoint has a home in the SGLang model, and nothing is unloaded.

Reads only the safetensors headers and builds no model: each checkpoint name goes through
``Metis16ForCausalLM.map_checkpoint_name`` and must land on a parameter name the SGLang model
defines (listed from a meta-device build would need SGLang's distributed setup, so the names
are checked against the module layout the mapping promises instead), and the per-pass layer
ids SGLang will size its caches by must cover every (pass, layer) once.

Set METIS_MODEL_DIR to a release directory (BF16 or FP8).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import model_dir  # noqa: E402

MODEL_DIR = model_dir(need_weights=True)
if MODEL_DIR is None or not list(MODEL_DIR.glob("*.safetensors")):
    pytest.skip("no release checkpoint: set METIS_MODEL_DIR", allow_module_level=True)

from sglang_metis.config import Metis16Config  # noqa: E402
from sglang_metis.model import Metis16ForCausalLM  # noqa: E402


def checkpoint_names(directory: Path) -> list[str]:
    names = []
    for path in sorted(directory.glob("*.safetensors")):
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(size))
        header.pop("__metadata__", None)
        names += list(header)
    return names


@pytest.fixture(scope="module")
def config():
    raw = json.loads((MODEL_DIR / "config.json").read_text())
    raw.pop("auto_map", None)
    return Metis16Config(**raw)


def test_every_tensor_maps(config):
    stub = SimpleNamespace(config=config)
    mapped = {}
    for name in checkpoint_names(MODEL_DIR):
        target = Metis16ForCausalLM.map_checkpoint_name(stub, name)
        if target is None:
            assert name.endswith((".k_router.weight", ".k_router.bias", ".k_budget.multiplier")) or name in (
                "depth_budget.multiplier", "lm_head.weight"
            ), name
            continue
        assert target.startswith("model."), (name, target)
        assert target not in mapped, f"{name} and {mapped.get(target)} both map to {target}"
        mapped[target] = name
    mamba = [t for t in mapped if ".mixer.mamba." in t]
    assert mamba and not any(".impl." in t for t in mapped)
    assert any(t.endswith(".mixer.mamba.A") for t in mamba)
    assert any(t.endswith(".experts.w13_weight") for t in mapped)
    assert any(t.endswith(".experts.w2_weight") for t in mapped)


def test_layer_ids(config):
    physical = config.num_physical_layers
    passes = config.num_passes
    assert config.num_hidden_layers == physical * passes
    attention = set(config.full_attention_layer_ids)
    mamba = set(config.mamba_layer_ids)
    assert not attention & mamba
    assert attention | mamba == set(range(physical * passes))
    for p in range(passes):
        for i in config.attention_layer_indices:
            assert config.slot(p, i) in attention
    assert len(attention) == passes * len(config.attention_layer_indices)
