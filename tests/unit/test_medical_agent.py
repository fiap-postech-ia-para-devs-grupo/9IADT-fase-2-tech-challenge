from __future__ import annotations

from typing import Any, cast

from tech_challenge.llm.medical_agent import DiagnosisInput, MedicalDiagnosisAgent, gerar_explicacao


class FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class FakeModels:
    def __init__(self, text: str | None) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(self.text)


class FakeClient:
    def __init__(self, text: str | None) -> None:
        self.models = FakeModels(text)


def _agent_with_response(text: str | None) -> MedicalDiagnosisAgent:
    agent = MedicalDiagnosisAgent.__new__(MedicalDiagnosisAgent)
    agent.model_name = "fake-model"
    agent.client = cast(Any, FakeClient(text))
    return agent


def test_build_prompt_orders_top_features_by_absolute_impact() -> None:
    agent = _agent_with_response("{}")

    prompt = agent._build_prompt(
        DiagnosisInput(
            features={"small": 0.1, "large_negative": -5.0, "large_positive": 4.0},
            prediction=1,
            probability=0.876,
            model_name="AG-test",
        )
    )

    assert "Predição: MALIGNO" in prompt
    assert "Probabilidade: 87.60 %" in prompt
    assert prompt.index("- large_negative: -5.0000") < prompt.index("- large_positive: 4.0000")


def test_clean_json_extracts_object_from_markdown_and_surrounding_text() -> None:
    agent = _agent_with_response("{}")

    assert agent._clean_json('texto ```json\n{"ok": true}\n``` fim') == '{"ok": true}'


def test_explain_returns_json_or_error_payload() -> None:
    valid_agent = _agent_with_response('{"interpretacao": "ok"}')
    invalid_agent = _agent_with_response("sem json")
    inp = DiagnosisInput(features={"radius": 1.0}, prediction=0, probability=0.7)

    assert valid_agent.explain(inp) == {"interpretacao": "ok"}
    assert invalid_agent.explain(inp) == {"error": "json_invalid", "raw": "sem json"}


def test_chat_returns_json_when_valid_and_raw_answer_when_invalid() -> None:
    valid_agent = _agent_with_response('{"resposta": "ok"}')
    invalid_agent = _agent_with_response("texto livre")

    assert valid_agent.chat("Pergunta?", context={"x": 1}) == {"resposta": "ok"}
    assert invalid_agent.chat("Pergunta?") == {"resposta": "texto livre"}


def test_gerar_explicacao_formats_external_contract() -> None:
    assert gerar_explicacao(1, 0.91, {"radius": 2}) == {
        "predicao": "maligno",
        "confianca": 0.91,
        "top_features": [{"nome": "radius", "valor": 2.0}],
    }
