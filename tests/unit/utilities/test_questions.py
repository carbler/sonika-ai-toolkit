"""Unit tests for the structured user-questions contract (utilities.questions).

Covers the shared schema, payload normalization, and human-readable summary
used by OrchestratorBot for the ask_user feature.
"""

from sonika_ai_toolkit.utilities.questions import (
    AskUserSchema,
    questions_to_payload,
    questions_summary,
)


_ASK_ARGS = {
    "reason": "Necesito datos para continuar",
    "questions": [
        {
            "id": "color",
            "text": "¿Qué color prefieres?",
            "type": "single_choice",
            "options": [
                {"value": "r", "label": "Rojo"},
                {"value": "b", "label": "Azul"},
            ],
            "required": True,
        }
    ],
}


class TestQuestionsContract:
    def test_schema_validates_and_normalizes(self):
        payload = questions_to_payload(_ASK_ARGS)
        assert payload["reason"] == _ASK_ARGS["reason"]
        assert payload["questions"][0]["id"] == "color"
        assert payload["questions"][0]["options"][0]["label"] == "Rojo"

    def test_schema_defaults(self):
        parsed = AskUserSchema.model_validate({"questions": [{"id": "n", "text": "Nombre?"}]})
        q = parsed.questions[0]
        assert q.type == "text"
        assert q.required is True
        assert q.options is None

    def test_summary_is_human_readable(self):
        text = questions_summary(questions_to_payload(_ASK_ARGS))
        assert "Necesito datos" in text
        assert "¿Qué color prefieres?" in text
        assert "Rojo" in text
