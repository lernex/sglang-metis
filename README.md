# sglang-metis

SGLang support for Lernex's Metis-1.6 models. Install it next to SGLang and
`Lernex/Metis-1.6-Logos` and `Lernex/Metis-1.6-Praxis` serve like any other model:
OpenAI-compatible API, reasoning split out of the answer, tool calls parsed and typed, radix
prefix caching, chunked prefill, CUDA graphs for prefill and decode.

Everything registers through SGLang's `sglang.srt.plugins` entry point, so there is nothing to
import or patch:

* `Metis16ForCausalLM`, the model, built on SGLang's own Mamba-2, fused MoE and radix
  attention kernels.
* `--reasoning-parser metis`, which separates the `<think>` block from the answer.
* `--tool-call-parser metis`, which turns `<|tool_call|><function=...>` blocks into OpenAI
  tool calls, typing each argument from the request's JSON schema.

## Install

```bash
pip install sglang-metis
```

It needs SGLang 0.5.22 or newer.

## Serve

```bash
python -m sglang.launch_server --model-path Lernex/Metis-1.6-Praxis \
    --reasoning-parser metis --tool-call-parser metis
```

No `--trust-remote-code`: the plugin brings its own configuration class and model, and never
runs the repository's Transformers code.

**Memory.** Praxis (3.0B) runs in BF16 on a 16 GB card. Logos (10.2B) is 20.8 GB in BF16; on a
16 or 24 GB card serve the FP8 checkpoint, which SGLang reads directly:

```bash
python -m sglang.launch_server --model-path Lernex/Metis-1.6-Logos-FP8 \
    --reasoning-parser metis --tool-call-parser metis
```

`Metis-1.6-Logos-FP8` and `Metis-1.6-Praxis-FP8` are compressed-tensors FP8 checkpoints (11.2
GB and 3.5 GB): FP8 weights with one scale per output channel and activations quantized per
token, for the routed experts, the Mamba input projections and the attention projections.
Everything the architecture is sensitive to stays as trained: the Mamba output projection,
the MoE latent projections, the shared expert, the routers, the depth memory and the
embedding. `scripts/make_fp8.py` writes them from a BF16 checkpoint. `--quantization fp8`
on the BF16 weights also works (the plugin then keeps the Mamba output projection in BF16
itself), but SGLang loads the whole BF16 model before quantizing it, so Logos needs a card
that holds 21 GB to do it that way.

**Concurrency.** Every request holds a Mamba state for each pass of each Mamba layer (20 per
request on Praxis, 32 on Logos, in FP32 as the model was trained), so the Mamba cache, not the
KV cache, sets how many requests run at once. SGLang's defaults size it for models with a
quarter of the state. On a 16 GB card these hold the most streams:

| Model | Flags |
|---|---|
| Praxis, BF16 or FP8 | `--max-mamba-cache-size 160 --max-total-tokens 131072 --cuda-graph-max-bs-decode 32` (53 requests) |
| Logos, FP8 | `--max-mamba-cache-size 24 --max-total-tokens 32768 --cuda-graph-max-bs-decode 8 --mem-fraction-static 0.92` (8 requests) |

`--mamba-ssm-dtype bfloat16` halves the state and doubles the requests, at a small cost in
accuracy the reference never pays.

The sampling defaults come from the model's `generation_config.json` (temperature 0.6,
top-p 0.95, top-k 20). Thinking is on unless a request turns it off with
`"chat_template_kwargs": {"enable_thinking": false}`.

## Use it

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

# Thinking mode (the default): the reasoning arrives separately from the answer.
reply = client.chat.completions.create(
    model="Lernex/Metis-1.6-Praxis",
    messages=[{"role": "user", "content": "What is 17 x 23?"}],
)
print(reply.choices[0].message.reasoning_content)
print(reply.choices[0].message.content)

# Tools, in direct mode (no thinking).
reply = client.chat.completions.create(
    model="Lernex/Metis-1.6-Praxis",
    messages=[{"role": "user", "content": "Weather in Lisbon for the next 3 days?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"],
            },
        },
    }],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(reply.choices[0].message.tool_calls)  # get_weather(city="Lisbon", days=3)
```

Streaming works the same way: reasoning, content and tool-call arguments arrive as deltas.

## How the model maps onto SGLang

Metis-1.6 runs its stack of blocks twice per token with shared weights. Each block carries
four residual streams joined by pass-conditioned Sinkhorn mixes, runs a Mamba-2 or
position-free GQA mixer and then a latent mixture of experts, and keeps a per-token depth
memory written after every attention layer and at the end of every pass.

Only the mixers look across positions, so the plugin gives each (pass, layer) pair its own
SGLang layer id, `pass * layers + layer`, while the weights exist once. SGLang's hybrid
backend then allocates a Mamba state or a KV cache per pass, and radix caching, chunked
prefill and CUDA graphs work as they do for any hybrid model. A Mamba block calls SGLang's
Mamba-2 backend with the pass's layer id; inside a CUDA graph that call is a graph break, as
SGLang's own hybrid models do it, because it reads each step's batch layout. An attention
block keeps one weightless `RadixAttention` per pass.

The parts the model is sensitive to stay in FP32 where the reference keeps them: the routers
and their balance bias, `A`, `D` and `dt_bias`, the SSM state, the Sinkhorn mixes, the depth
memory's attention and gates, the depth head and every RMSNorm. The 4x4 mixes and their read
and write weights depend only on the pass, so they are computed once when the weights load.

## Status

Measured on an RTX 5070 Ti (16 GB) with SGLang 0.5.22, CUDA graphs on, against FP32 logits
from the native training runtime on the release's reference prompts (a 16-token thinking
chat, a 176-token chat with tools, 300 tokens of prose). Mean top-20 KL to the reference, and
how often the top token agrees:

| Model | Weights | Thinking chat | Chat with tools | Prose | tok/s, 1 / 32 streams |
|---|---|---|---|---|---|
| Praxis | BF16 | 0.0013, 100% | 0.014, 94.9% | 0.0018, 98.7% | 153 / 1969 |
| Praxis | FP8 checkpoint | 0.0055, 100% | 0.036, 93.7% | 0.0077, 96.0% | 153 / 2096 |
| Praxis | BF16, `--quantization fp8` | 0.011, 93.3% | 0.042, 89.1% | 0.0075, 98.0% | 152 / 2098 |
| Logos | FP8 checkpoint | 0.0055, 93.3% | 0.033, 91.4% | 0.0079, 95.3% | 82 / 430 (8 streams) |

Praxis in BF16 is inside the noise floor of the native model run in BF16 (0.0073, 0.014,
0.0022). In every configuration the thinking and direct answers are correct.

* `scripts/smoke.py` produced the table: prompt logprobs on fixed token sequences, greedy
  generation in thinking and direct mode, one stream and a batch.
* `scripts/serve_smoke.py` against `sglang.launch_server` with both parsers: thinking-mode
  reasoning arrives as `reasoning_content`, a tool call comes back typed (`days` as the
  integer 3) in direct and thinking mode, the streamed answers equal the whole ones, and the
  model answers from the tool's result on the next turn.
* `tests/` checks both parsers on outputs rendered by the real tokenizer and chat template,
  whole and streamed in deltas of 1, 2, 3, 7 and all characters, and that every checkpoint
  tensor (BF16 and FP8) maps onto the model.

## Limitations

* Tensor parallelism must divide the Mamba groups and KV heads: 10 on Logos (TP 1, 2, 5 or
  10), 8 on Praxis (TP 1, 2, 4 or 8). Only TP 1 has been run. Pipeline parallelism is not
  supported: every layer runs twice per token.
* Speculative decoding and LoRA adapters are untested.
* The adaptive-depth and adaptive-width heads in the checkpoint are not used: the release
  runs the fixed policy (2 passes, top 4), which is what this plugin implements.
* SGLang ships no tuned fused-MoE kernel configuration for these expert shapes on most cards
  and falls back to its default; tuning one with SGLang's `tuning_fused_moe_triton.py` may
  add speed.

## Tests

```bash
pip install -e ".[test]"
METIS_MODEL_DIR=/path/to/Metis-1.6-Praxis pytest
```

Without `METIS_MODEL_DIR` the parser tests fetch the tokenizer from the Hub and the
weight-map tests skip.

## License

Apache 2.0.
