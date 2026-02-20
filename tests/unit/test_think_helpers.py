"""
Unit tests for sonika_ai_toolkit.agents.think module-level helpers and
ThinkBot static methods.

Covers:
  - _get_text_content(): string passthrough, list format (Gemini), edge cases
  - extract_thinking(): Gemini list format, additional_kwargs (reasoning_content,
    thinking), <think> tag fallback, graceful None on missing data
  - ThinkBot._strip_think_tags(): removal, nested tags safety, multiline
  - ThinkBot._clean_content(): combined pipeline
  - ThinkBot._extract_token_usage(): callback path, usage_metadata fallback
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

try:
    from sonika_ai_toolkit.agents.think import (
        _get_text_content,
        extract_thinking,
        ThinkBot,
    )
except ImportError:
    import pytest as _pytest
    _pytest.skip("sonika_ai_toolkit.agents.think not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# _get_text_content
# ---------------------------------------------------------------------------

class TestGetTextContent:
    def test_plain_string_returned_as_is(self):
        assert _get_text_content("hello world") == "hello world"

    def test_empty_string_returned_as_is(self):
        assert _get_text_content("") == ""

    def test_none_returns_empty_string(self):
        assert _get_text_content(None) == ""

    def test_list_with_only_thinking_parts_returns_empty(self):
        content = [{"type": "thinking", "thinking": "some reasoning"}]
        result = _get_text_content(content)
        assert result == ""

    def test_list_with_plain_string_parts(self):
        content = [{"type": "thinking", "thinking": "reasoning"}, "final answer"]
        result = _get_text_content(content)
        assert result == "final answer"

    def test_list_with_text_type_dict(self):
        content = [
            {"type": "thinking", "thinking": "reasoning"},
            {"type": "text", "text": "The answer is 42"},
        ]
        result = _get_text_content(content)
        assert result == "The answer is 42"

    def test_list_multiple_text_parts_joined(self):
        content = ["part one", "part two"]
        result = _get_text_content(content)
        assert result == "part one\npart two"

    def test_non_string_non_list_is_stringified(self):
        result = _get_text_content(123)
        assert result == "123"

    def test_list_with_content_key_in_dict(self):
        content = [{"type": "output", "content": "answer here"}]
        result = _get_text_content(content)
        assert result == "answer here"


# ---------------------------------------------------------------------------
# extract_thinking
# ---------------------------------------------------------------------------

class TestExtractThinking:
    def test_gemini_list_content_thinking_part(self):
        msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "I need to reason about this"},
                "The answer is 42",
            ]
        )
        result = extract_thinking(msg)
        assert result == "I need to reason about this"

    def test_gemini_list_content_no_thinking_part(self):
        msg = AIMessage(content=[{"type": "text", "text": "just an answer"}])
        result = extract_thinking(msg)
        assert result is None

    def test_deepseek_reasoning_content_in_additional_kwargs(self):
        msg = AIMessage(
            content="final answer",
            additional_kwargs={"reasoning_content": "DeepSeek thought process"},
        )
        result = extract_thinking(msg)
        assert result == "DeepSeek thought process"

    def test_gemini_thinking_string_in_additional_kwargs(self):
        msg = AIMessage(
            content="final answer",
            additional_kwargs={"thinking": "Gemini thought process"},
        )
        result = extract_thinking(msg)
        assert result == "Gemini thought process"

    def test_gemini_thinking_list_in_additional_kwargs(self):
        msg = AIMessage(
            content="answer",
            additional_kwargs={"thinking": ["part1", "part2"]},
        )
        result = extract_thinking(msg)
        assert result == "part1\npart2"

    def test_fallback_think_tags_extracted(self):
        msg = AIMessage(content="<think>step by step reasoning</think>final answer")
        result = extract_thinking(msg)
        assert result == "step by step reasoning"

    def test_fallback_think_tags_multiline(self):
        msg = AIMessage(content="<think>\nline one\nline two\n</think>done")
        result = extract_thinking(msg)
        assert "line one" in result
        assert "line two" in result

    def test_no_thinking_returns_none(self):
        msg = AIMessage(content="plain response with no thinking")
        result = extract_thinking(msg)
        assert result is None

    def test_empty_content_returns_none(self):
        msg = AIMessage(content="")
        result = extract_thinking(msg)
        assert result is None

    def test_empty_think_tag_returns_none(self):
        msg = AIMessage(content="<think>   </think>answer")
        result = extract_thinking(msg)
        assert result is None

    def test_empty_reasoning_content_returns_none(self):
        msg = AIMessage(
            content="answer",
            additional_kwargs={"reasoning_content": ""},
        )
        result = extract_thinking(msg)
        assert result is None

    def test_list_thinking_uses_text_key_as_fallback(self):
        """Gemini may use 'text' key instead of 'thinking' for the content."""
        msg = AIMessage(
            content=[{"type": "thinking", "text": "alternative key text"}]
        )
        result = extract_thinking(msg)
        assert result == "alternative key text"


# ---------------------------------------------------------------------------
# ThinkBot._strip_think_tags
# ---------------------------------------------------------------------------

class TestStripThinkTags:
    def test_removes_think_block(self):
        text = "<think>some thoughts</think>final answer"
        result = ThinkBot._strip_think_tags(text)
        assert result == "final answer"

    def test_removes_multiline_think_block(self):
        text = "<think>\nthinking\nmore thinking\n</think>answer"
        result = ThinkBot._strip_think_tags(text)
        assert result == "answer"

    def test_no_think_tag_returns_unchanged(self):
        text = "no thinking here"
        result = ThinkBot._strip_think_tags(text)
        assert result == "no thinking here"

    def test_empty_string(self):
        result = ThinkBot._strip_think_tags("")
        assert result == ""

    def test_strips_surrounding_whitespace(self):
        result = ThinkBot._strip_think_tags("  <think>x</think>  ")
        assert result == ""

    def test_multiple_think_blocks(self):
        text = "<think>first</think>middle<think>second</think>end"
        result = ThinkBot._strip_think_tags(text)
        assert "first" not in result
        assert "second" not in result
        assert "middle" in result
        assert "end" in result


# ---------------------------------------------------------------------------
# ThinkBot._clean_content
# ---------------------------------------------------------------------------

class TestCleanContent:
    def test_plain_string_strips_think_tags(self):
        result = ThinkBot._clean_content("<think>reasoning</think>answer")
        assert result == "answer"

    def test_gemini_list_content_extracts_text_only(self):
        content = [
            {"type": "thinking", "thinking": "private reasoning"},
            "public answer",
        ]
        result = ThinkBot._clean_content(content)
        assert result == "public answer"
        assert "private reasoning" not in result

    def test_plain_string_no_tags_returned_as_is(self):
        result = ThinkBot._clean_content("direct answer")
        assert result == "direct answer"

    def test_empty_list_returns_empty_string(self):
        result = ThinkBot._clean_content([])
        assert result == ""


# ---------------------------------------------------------------------------
# ThinkBot._extract_token_usage
# ---------------------------------------------------------------------------

class TestExtractTokenUsage:
    def _make_state(self, messages=None):
        return {"messages": messages or []}

    def test_uses_callback_when_total_tokens_positive(self):
        cb = MagicMock()
        cb.total_tokens = 100
        cb.prompt_tokens = 60
        cb.completion_tokens = 40

        result = ThinkBot._extract_token_usage(self._make_state(), cb)

        assert result == {
            "prompt_tokens": 60,
            "completion_tokens": 40,
            "total_tokens": 100,
        }

    def test_falls_back_to_usage_metadata_when_callback_zero(self):
        cb = MagicMock()
        cb.total_tokens = 0
        cb.prompt_tokens = 0
        cb.completion_tokens = 0

        msg = AIMessage(content="answer")
        msg.usage_metadata = {"input_tokens": 50, "output_tokens": 30}

        result = ThinkBot._extract_token_usage(self._make_state([msg]), cb)

        assert result["prompt_tokens"] == 50
        assert result["completion_tokens"] == 30
        assert result["total_tokens"] == 80

    def test_sums_multiple_messages_metadata(self):
        cb = MagicMock()
        cb.total_tokens = 0
        cb.prompt_tokens = 0
        cb.completion_tokens = 0

        msg1 = AIMessage(content="first")
        msg1.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        msg2 = AIMessage(content="second")
        msg2.usage_metadata = {"input_tokens": 20, "output_tokens": 10}

        result = ThinkBot._extract_token_usage(self._make_state([msg1, msg2]), cb)

        assert result["prompt_tokens"] == 30
        assert result["completion_tokens"] == 15
        assert result["total_tokens"] == 45

    def test_no_metadata_returns_zeros(self):
        cb = MagicMock()
        cb.total_tokens = 0
        cb.prompt_tokens = 0
        cb.completion_tokens = 0

        msg = AIMessage(content="no metadata")
        result = ThinkBot._extract_token_usage(self._make_state([msg]), cb)

        assert result == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_empty_messages_returns_zeros(self):
        cb = MagicMock()
        cb.total_tokens = 0
        cb.prompt_tokens = 0
        cb.completion_tokens = 0

        result = ThinkBot._extract_token_usage(self._make_state([]), cb)

        assert result == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
