"""Unit tests for DynamicToolSynthesizer (tools.synthesizer).

Focus on the pure helpers and the end-to-end synthesize() path with a mocked
code_model, so no real LLM call and no leftover files outside tmp_path.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.tools import BaseTool

from sonika_ai_toolkit.tools.synthesizer import (
    DynamicToolSynthesizer,
    _extract_code_block,
    _to_class_name,
)


class TestToClassName:
    def test_snake_to_pascal_with_suffix(self):
        assert _to_class_name("weather_lookup") == "WeatherLookupTool"

    def test_single_word(self):
        assert _to_class_name("ping") == "PingTool"


class TestExtractCodeBlock:
    def test_python_fenced_block(self):
        text = "blah\n```python\nprint('hi')\n```\ntrailing"
        assert _extract_code_block(text) == "print('hi')"

    def test_plain_fenced_block_fallback(self):
        text = "```\nx = 1\n```"
        assert _extract_code_block(text) == "x = 1"

    def test_no_block_returns_none(self):
        assert _extract_code_block("just prose, no code") is None


_GENERATED_TOOL = '''```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class EchoArgs(BaseModel):
    text: str


class GreetTool(BaseTool):
    name: str = "greet"
    description: str = "Echo the text back."
    args_schema: type = EchoArgs

    def _run(self, text: str) -> str:
        return text
```'''


class TestSynthesize:
    @pytest.mark.asyncio
    async def test_generates_and_loads_tool(self, tmp_path):
        code_model = MagicMock()
        code_model.model = MagicMock()
        code_model.model.ainvoke = AsyncMock(
            return_value=MagicMock(content=_GENERATED_TOOL)
        )
        synth = DynamicToolSynthesizer(code_model, skills_dir=str(tmp_path))

        tool = await synth.synthesize("greet", "Echo the text back")

        assert isinstance(tool, BaseTool)
        assert tool.name == "greet"
        assert tool._run(text="hola") == "hola"
        assert (tmp_path / "greet.py").exists()

    @pytest.mark.asyncio
    async def test_raises_when_no_code_block(self, tmp_path):
        code_model = MagicMock()
        code_model.model = MagicMock()
        code_model.model.ainvoke = AsyncMock(
            return_value=MagicMock(content="sorry, no code here")
        )
        synth = DynamicToolSynthesizer(code_model, skills_dir=str(tmp_path))

        with pytest.raises(RuntimeError):
            await synth.synthesize("greet", "whatever")
