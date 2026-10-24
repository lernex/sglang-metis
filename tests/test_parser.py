# SPDX-License-Identifier: Apache-2.0
"""The metis reasoning and tool-call parsers on outputs rendered by the real chat template.

Each case is an assistant turn written as a message. The real tokenizer and chat template
render the conversation, and the part after the generation prompt (minus the stop token) is
what the model would have produced. It goes through the reasoning parser and then the tool
parser, as SGLang's chat endpoint runs them for ``--reasoning-parser metis --tool-call-parser
metis``, whole and streamed in deltas of several sizes, and must give back the message.

Set METIS_MODEL_DIR to a directory holding the Metis tokenizer and chat template.
"""

from __future__ import annotations

import json

import pytest

from conftest import model_dir  # noqa: E402

_dir = model_dir()
if _dir is None or not (_dir / "tokenizer.json").exists():
    pytest.skip("no Metis tokenizer: set METIS_MODEL_DIR or log in to the Hub", allow_module_level=True)
MODEL_DIR = str(_dir)

import sglang_metis  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import Tool  # noqa: E402
from sglang.srt.function_call.function_call_parser import FunctionCallParser  # noqa: E402
from sglang.srt.parser.reasoning_parser import ReasoningParser  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

sglang_metis.register()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Run a snippet.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}, "language": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search an index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {"type": "object"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "id": {"type": "string"},
                },
            },
        },
    },
]

USER = {"role": "user", "content": "Help me with a few things."}

CASES = {
    "think_content": dict(
        thinking=True,
        reasoning="Two plus two.\nThat is 4, and 4 > 3.",
        content="The answer is 4.",
        calls=[],
    ),
    "think_one_call": dict(
        thinking=True,
        reasoning="The user wants the weather in Paris for three days.",
        content="Let me check the forecast.",
        calls=[("get_weather", {"city": "Paris", "days": 3})],
    ),
    "think_calls_no_content": dict(
        thinking=True,
        reasoning="I need two tools. First <weather>, then code: if a < b and b > c.",
        content="",
        calls=[
            ("get_weather", {"city": "New York", "days": 2, "metric": True}),
            (
                "run_code",
                {"code": "def f(x):\n    return 1 < x < 9\n\nprint(f(3))", "language": "python"},
            ),
        ],
    ),
    "direct_content": dict(
        thinking=False,
        reasoning="",
        content="Sure: <b>bold</b>, a café and an arrow → done.",
        calls=[],
    ),
    "direct_typed_call": dict(
        thinking=False,
        reasoning="",
        content="",
        calls=[
            (
                "search",
                {
                    "query": "SGLang plugins",
                    "filters": {"lang": "en", "year": 2026},
                    "tags": ["ai", "llm"],
                    "limit": 5,
                    "ratio": 0.5,
                    "id": "0042",
                },
            )
        ],
    ),
    "think_content_then_call": dict(
        thinking=True,
        reasoning="Search first.",
        content="I will search for that.\nOne moment.",
        calls=[("search", {"query": "a </b> tag", "limit": 1})],
    ),
}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def assistant_message(case):
    message = {"role": "assistant", "content": case["content"]}
    if case["reasoning"]:
        message["reasoning_content"] = case["reasoning"]
    if case["calls"]:
        message["tool_calls"] = [
            {"type": "function", "function": {"name": n, "arguments": a}} for n, a in case["calls"]
        ]
    return message


def render(tokenizer, case):
    """The output text for the case, exactly as the template has it, without the stop token."""
    kwargs = dict(tools=TOOLS, enable_thinking=case["thinking"], tokenize=False)
    prompt = tokenizer.apply_chat_template([USER], add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template([USER, assistant_message(case)], **kwargs)
    assert full.startswith(prompt)
    assert full.endswith("<|fim_suffix|>")
    return full[len(prompt) : -len("<|fim_suffix|>")]


def tools():
    return [Tool.model_validate(t) for t in TOOLS]


def reasoner(case, stream: bool):
    # Thinking mode starts inside the reasoning (the prompt opened it); direct mode has none,
    # and SGLang does not run the reasoning parser on it.
    if not case["thinking"]:
        return None
    return ReasoningParser("metis", stream_reasoning=stream, force_reasoning=True)


def expected(case):
    return case["reasoning"] or None, case["content"] or None, list(case["calls"])


def calls_of(items):
    return [(c.name, json.loads(c.parameters)) for c in items]


@pytest.mark.parametrize("name", list(CASES))
def test_whole_output(tokenizer, name):
    case = CASES[name]
    text = render(tokenizer, case)
    reasoning, content = None, text
    if (r := reasoner(case, stream=False)) is not None:
        reasoning, content = r.parse_non_stream(text)
    parser = FunctionCallParser(tools(), "metis")
    calls = []
    if parser.has_tool_call(content):
        content, items = parser.parse_non_stream(content)
        calls = calls_of(items)
    reasoning = reasoning.strip() if reasoning else None
    assert (reasoning, content.strip() or None, calls) == expected(case)


@pytest.mark.parametrize("step", [1, 2, 3, 7, 10_000])
@pytest.mark.parametrize("name", list(CASES))
def test_streaming(tokenizer, name, step):
    case = CASES[name]
    text = render(tokenizer, case)
    r = reasoner(case, stream=True)
    parser = FunctionCallParser(tools(), "metis")
    reasoning, content, slots = [], [], {}
    for start in range(0, len(text), step):
        delta = text[start : start + step]
        if r is not None:
            thought, delta = r.parse_stream_chunk(delta)
            reasoning.append(thought or "")
        if not delta:
            continue
        normal, items = parser.parse_stream_chunk(delta)
        content.append(normal or "")
        for item in items:
            slot = slots.setdefault(item.tool_index, {"name": "", "args": ""})
            slot["name"] += item.name or ""
            slot["args"] += item.parameters or ""
    calls = [(s["name"], json.loads(s["args"] or "{}")) for _, s in sorted(slots.items())]
    got = ("".join(reasoning).strip() or None, "".join(content).strip() or None, calls)
    assert got == expected(case)


def test_registered_names():
    from sglang_metis.parser import MetisDetector, MetisReasoningDetector

    assert ReasoningParser.DetectorMap["metis"] is MetisReasoningDetector
    assert FunctionCallParser.ToolCallParserEnum["metis"] is MetisDetector
