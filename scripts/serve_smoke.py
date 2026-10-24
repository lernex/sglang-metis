# SPDX-License-Identifier: Apache-2.0
"""Talk to a running SGLang server of Metis-1.6 through the OpenAI client.

Start the server first::

    python -m sglang.launch_server --model-path Lernex/Metis-1.6-Praxis \\
        --reasoning-parser metis --tool-call-parser metis

then run ``python scripts/serve_smoke.py``. It sends a thinking-mode chat and a direct-mode
tool request, each once whole and once streamed, checks that the streamed answer adds up to
the whole one, then answers the tool call and checks the model uses the result.
"""

from __future__ import annotations

import argparse
import json
import sys

from openai import OpenAI

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        },
    }
]


def _reasoning(obj):
    return getattr(obj, "reasoning_content", None) or getattr(obj, "reasoning", None)


def whole(client, model, **kwargs):
    message = client.chat.completions.create(model=model, **kwargs).choices[0].message
    calls = [(c.function.name, json.loads(c.function.arguments)) for c in message.tool_calls or []]
    return _reasoning(message), message.content, calls, message


def streamed(client, model, **kwargs):
    reasoning, content, calls = [], [], {}
    for chunk in client.chat.completions.create(model=model, stream=True, **kwargs):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if _reasoning(delta):
            reasoning.append(_reasoning(delta))
        if delta.content:
            content.append(delta.content)
        for call in delta.tool_calls or []:
            slot = calls.setdefault(call.index, {"name": "", "arguments": ""})
            if call.function and call.function.name:
                slot["name"] += call.function.name
            if call.function and call.function.arguments:
                slot["arguments"] += call.function.arguments
    return (
        "".join(reasoning) or None,
        "".join(content) or None,
        [(c["name"], json.loads(c["arguments"] or "{}")) for _, c in sorted(calls.items())],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:30000/v1")
    parser.add_argument("--model", default=None, help="served model name (default: the first)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    model = args.model or client.models.list().data[0].id
    ask = [{"role": "user", "content": "What's the weather in Lisbon for the next 3 days?"}]
    requests = {
        "thinking chat": dict(
            messages=[{"role": "user", "content": "What is 17 x 23? Answer briefly."}],
            max_tokens=args.max_tokens,
            temperature=0,
        ),
        "direct tool call": dict(
            messages=ask,
            tools=TOOLS,
            max_tokens=args.max_tokens,
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        "thinking tool call": dict(messages=ask, tools=TOOLS, max_tokens=args.max_tokens, temperature=0),
    }
    failures = 0
    last = None
    for name, request in requests.items():
        a = whole(client, model, **request)
        b = streamed(client, model, **request)
        same = a[0] == b[0] and (a[1] or "").strip() == (b[1] or "").strip() and a[2] == b[2]
        print(f"== {name}: streamed matches whole: {same}")
        print(f"   reasoning: {a[0]!r:.300}")
        print(f"   content:   {a[1]!r:.300}")
        print(f"   calls:     {a[2]}")
        failures += not same
        if name == "direct tool call":
            failures += a[0] is not None  # direct mode never reports reasoning
        if "tool" in name:
            failures += not (a[2] and a[2][0][0] == "get_weather")
            last = a[3]

    if last is not None and last.tool_calls:
        call = last.tool_calls[0]
        result = {"city": "Lisbon", "forecast": ["sunny 24C", "cloudy 21C", "rain 18C"]}
        followup = ask + [
            {"role": "assistant", "content": last.content or "", "tool_calls": [call.model_dump()]},
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)},
        ]
        r, content, calls, _ = whole(
            client, model, messages=followup, tools=TOOLS, max_tokens=args.max_tokens, temperature=0
        )
        used = bool(content) and any(w in content.lower() for w in ("sunny", "rain", "24"))
        print(f"== tool result round trip: answer uses the result: {used}")
        print(f"   reasoning: {r!r:.300}")
        print(f"   content:   {content!r:.400}")
        failures += not used
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
