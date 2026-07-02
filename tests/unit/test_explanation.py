from __future__ import annotations

import pytest

from tech_challenge.diagnosis import FeatureImpact
from tech_challenge.explanation import (
    DISCLAIMER,
    LLMConfigurationError,
    LLMProviderError,
    chat_about_diagnosis,
    explain_diagnosis,
)


class FakeAgent:
    response: dict = {"interpretacao": "Explicacao", "disclaimer": "Aviso"}

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def explain(self, inp) -> dict:
        return self.response

    def chat(self, question: str, context: dict | None = None) -> dict:
        return self.response


class RaisingAgent(FakeAgent):
    def explain(self, inp) -> dict:
        raise RuntimeError("boom")

    def chat(self, question: str, context: dict | None = None) -> dict:
        raise RuntimeError("boom")


def test_explain_diagnosis_normalizes_mapping_features_and_uses_default_disclaimer(monkeypatch) -> None:
    FakeAgent.response = {"resposta": "Texto da explicacao"}
    monkeypatch.setattr("tech_challenge.explanation.MedicalDiagnosisAgent", FakeAgent)

    result = explain_diagnosis(
        "MALIGNO",
        0.91,
        [{"feature": "radius", "value": "2.5", "impact": "1.2"}],
        api_key="key",
    )

    assert result.explanation == "Texto da explicacao"
    assert result.disclaimer == DISCLAIMER
    assert result.details == {"resposta": "Texto da explicacao"}


def test_explain_diagnosis_accepts_feature_impact_objects(monkeypatch) -> None:
    FakeAgent.response = {"interpretacao": "Objeto", "disclaimer": "Aviso"}
    monkeypatch.setattr("tech_challenge.explanation.MedicalDiagnosisAgent", FakeAgent)

    result = explain_diagnosis("BENIGNO", 0.82, [FeatureImpact("texture", 3.0, -1.0)], api_key="key")

    assert result.explanation == "Objeto"
    assert result.disclaimer == "Aviso"


def test_explain_diagnosis_converts_agent_error_payload_to_provider_error(monkeypatch) -> None:
    FakeAgent.response = {"error": "json_invalid"}
    monkeypatch.setattr("tech_challenge.explanation.MedicalDiagnosisAgent", FakeAgent)

    with pytest.raises(LLMProviderError, match="json_invalid"):
        explain_diagnosis("BENIGNO", 0.8, [], api_key="key")


def test_explain_diagnosis_wraps_provider_exception(monkeypatch) -> None:
    monkeypatch.setattr("tech_challenge.explanation.MedicalDiagnosisAgent", RaisingAgent)

    with pytest.raises(LLMProviderError, match="boom"):
        explain_diagnosis("BENIGNO", 0.8, [], api_key="key")


def test_chat_about_diagnosis_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        chat_about_diagnosis("   ", api_key="key")


def test_chat_about_diagnosis_extracts_nested_json_answer(monkeypatch) -> None:
    FakeAgent.response = {"resposta": '{"answer": "Resposta aninhada"}'}
    monkeypatch.setattr("tech_challenge.explanation.MedicalDiagnosisAgent", FakeAgent)

    result = chat_about_diagnosis("Pergunta?", context={"case": 1}, api_key="key")

    assert result.answer == "Resposta aninhada"


def test_chat_about_diagnosis_requires_answer(monkeypatch) -> None:
    FakeAgent.response = {}
    monkeypatch.setattr("tech_challenge.explanation.MedicalDiagnosisAgent", FakeAgent)

    with pytest.raises(LLMProviderError, match="did not include an answer"):
        chat_about_diagnosis("Pergunta?", api_key="key")


def test_chat_about_diagnosis_wraps_provider_exception(monkeypatch) -> None:
    monkeypatch.setattr("tech_challenge.explanation.MedicalDiagnosisAgent", RaisingAgent)

    with pytest.raises(LLMProviderError, match="boom"):
        chat_about_diagnosis("Pergunta?", api_key="key")


def test_google_api_key_is_required_when_not_provided(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("tech_challenge.explanation.load_dotenv", lambda: None)

    with pytest.raises(LLMConfigurationError, match="GOOGLE_API_KEY"):
        explain_diagnosis("BENIGNO", 0.8, [])
