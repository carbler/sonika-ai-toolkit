"""Unit tests for AskUserQuestionTool (tools.ask_user).

The tool is a zero-risk signalling tool: it carries the ask_user contract and
must never execute a real side effect.
"""

from sonika_ai_toolkit.tools.ask_user import AskUserQuestionTool
from sonika_ai_toolkit.utilities.questions import ASK_USER_TOOL_NAME, AskUserSchema


class TestAskUserQuestionTool:
    def test_metadata(self):
        tool = AskUserQuestionTool()
        assert tool.name == ASK_USER_TOOL_NAME
        assert tool.risk_level == 0
        assert tool.args_schema is AskUserSchema
