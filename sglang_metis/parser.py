# SPDX-License-Identifier: Apache-2.0
"""Reasoning and tool-call parsing for Metis-1.6 (chat template ``metis-chat/v1``).

An assistant turn, as the template renders it::

    <think>
    REASONING
    </think>
    CONTENT<|tool_call|><function=NAME>
    <parameter=P>
    VALUE
    </parameter>
    </function><|tool_call|><function=...>...</function><|fim_suffix|>

* Thinking mode (the default): the generation prompt already ends with ``<think>\\n``, so the
  output starts inside the reasoning, which ends at ``</think>``. Direct mode
  (``enable_thinking=false``) renders an empty think block, and every token is content.
* ``<|tool_call|>`` is a special token and ``<think>``/``</think>`` are plain text. Calls follow
  each other with nothing in between and have no closing wrapper; the turn ends with
  ``<|fim_suffix|>``, a stop token.
* A string parameter's VALUE is raw text; any other type is JSON, typed from the tool schema.

The call body is the Qwen3-Coder XML format, so the tool detector reuses SGLang's parser for it
and only changes how a call opens.
"""

from __future__ import annotations

import re

from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.parser.reasoning_parser import BaseReasoningFormatDetector

TOOL_CALL = "<|tool_call|>"


class MetisReasoningDetector(BaseReasoningFormatDetector):
    """``--reasoning-parser metis``"""

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=[TOOL_CALL, "<|fim_suffix|>", "<|endoftext|>"],
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            # A call straight out of the reasoning ends it implicitly.
            tool_start_token=TOOL_CALL,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            # The template opens the think block in the prompt, so the output carries no <think>.
            thinks_internally=True,
            reasoning_default="enable_thinking",
            force_nonempty_content=force_nonempty_content,
        )


class MetisDetector(Qwen3CoderDetector):
    """``--tool-call-parser metis``: Qwen3-Coder call bodies, opened by ``<|tool_call|>``, unwrapped."""

    def __init__(self):
        super().__init__()
        self.tool_call_start_token = TOOL_CALL
        # Calls are not wrapped, so nothing closes the tool region; this never occurs in output.
        self.tool_call_end_token = "<|metis_no_end|>"
        self.tool_call_regex = re.compile(
            re.escape(TOOL_CALL) + r"(.*?)(?=" + re.escape(TOOL_CALL) + r"|$)", re.DOTALL
        )

    def has_tool_call(self, text: str) -> bool:
        # With special tokens skipped the marker is gone and a call starts at <function=.
        return TOOL_CALL in text or self.tool_call_prefix in text

    def detect_and_parse(self, text, tools):
        if TOOL_CALL not in text and self.tool_call_prefix in text:
            start = text.find(self.tool_call_prefix)
            parsed = super().detect_and_parse(text[:start] + TOOL_CALL + text[start:], tools)
            return parsed
        return super().detect_and_parse(text, tools)

    def get_structural_tag_name(self) -> str:
        return "metis"
