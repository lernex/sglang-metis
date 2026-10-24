# SPDX-License-Identifier: Apache-2.0
"""Metis-1.6 in SGLang on real weights: teacher-forced parity, generation, throughput.

    python scripts/smoke.py --model <hf dir> --refs <dir with prompts.json, native_logits.pt>
        [--quantization fp8] [--json out.json]

Parity is SGLang's top-k prompt logprobs on fixed token sequences against FP32 reference logits,
as KL on SGLang's reported support (both renormalized there) and argmax agreement.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch


def topk_kl(top, ref_logprobs) -> float:
    """KL(ref || sglang) on SGLang's reported tokens, both renormalized on that support."""
    ids = [t for _, t, *_ in top]
    v = [lp for lp, *_ in top]
    r = [ref_logprobs[t].item() for t in ids]
    lv = math.log(sum(math.exp(x) for x in v))
    lr = math.log(sum(math.exp(x) for x in r))
    return sum(math.exp(rr - lr) * ((rr - lr) - (vv - lv)) for rr, vv in zip(r, v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--mem-fraction-static", type=float, default=0.80)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--disable-cuda-graph", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--engine-kw", action="append", default=[], metavar="KEY=VALUE",
                    help="extra sgl.Engine argument, e.g. cuda_graph_max_bs=32 (repeatable)")
    args = ap.parse_args()

    import sglang as sgl
    from transformers import AutoTokenizer

    refs = Path(args.refs)
    prompts = json.loads((refs / "prompts.json").read_text())
    native = torch.load(refs / "native_logits.pt", map_location="cpu")
    tok = AutoTokenizer.from_pretrained(args.model)

    t0 = time.time()
    engine = sgl.Engine(
        model_path=args.model,
        quantization=args.quantization,
        mem_fraction_static=args.mem_fraction_static,
        context_length=8192,
        disable_cuda_graph=args.disable_cuda_graph,
        **{"log_level": "warning", **{k: json.loads(v) for k, v in (kv.split("=", 1) for kv in args.engine_kw)}},
    )
    report = {"config": vars(args), "load_seconds": time.time() - t0, "parity": {}}

    for name, ids in prompts.items():
        out = engine.generate(
            input_ids=ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=args.topk,
        )
        tops = out["meta_info"]["input_top_logprobs"]
        ref = torch.log_softmax(native[name].float(), dim=-1)
        # input_top_logprobs[i] scores token i from the tokens before it; reference row i - 1.
        kls, agree = [], []
        for i in range(1, len(ids)):
            top = tops[i]
            if not top:
                continue
            kls.append(topk_kl(top, ref[i - 1]))
            agree.append(max(top, key=lambda x: x[0])[1] == int(ref[i - 1].argmax()))
        report["parity"][name] = {
            "tokens": len(ids),
            "mean_topk_kl": sum(kls) / len(kls),
            "argmax_agree": sum(agree) / len(agree),
        }
        print(name, report["parity"][name], flush=True)

    gen = {}
    for mode, thinking in (("thinking", True), ("direct", False)):
        text = tok.apply_chat_template(
            [{"role": "user", "content": "What is 17 x 23? Answer briefly."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
        )
        t = time.time()
        out = engine.generate(text, {"max_new_tokens": 1024, "temperature": 0.0})
        gen[mode] = {"seconds": time.time() - t, "tokens": out["meta_info"]["completion_tokens"], "text": out["text"][-300:]}
        print(mode, gen[mode], flush=True)
    report["generation"] = gen

    text = tok.apply_chat_template(
        [{"role": "user", "content": "Write a long story about a lighthouse keeper."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    sp = {"max_new_tokens": args.gen_tokens, "temperature": 0.0, "ignore_eos": True}
    t = time.time()
    one = engine.generate(text, sp)
    single = one["meta_info"]["completion_tokens"] / (time.time() - t)
    t = time.time()
    many = engine.generate([text] * args.batch, sp)
    batch = sum(o["meta_info"]["completion_tokens"] for o in many) / (time.time() - t)
    report["throughput"] = {"single_stream_tok_s": single, f"batch{args.batch}_tok_s": batch}
    print(report["throughput"], flush=True)
    engine.shutdown()
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
