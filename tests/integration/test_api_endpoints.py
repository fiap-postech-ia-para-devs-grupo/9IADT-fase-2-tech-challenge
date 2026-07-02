from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from tech_challenge.adapters import api
from tech_challenge.diagnosis import FeatureImpact

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class FakeDiagnosisResult:
    prediction: str
    confidence: float
    top_features: list[FeatureImpact]


client = TestClient(api.app)


def test_health_endpoint() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_diagnose_endpoint_serializes_domain_result(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "diagnose_patient",
        lambda patient_index: FakeDiagnosisResult(
            prediction="MALIGNO",
            confidence=0.93,
            top_features=[FeatureImpact("radius", 2.0, 1.4)],
        ),
    )

    response = client.post("/diagnose", json={"patient_index": 7})

    assert response.status_code == 200
    assert response.json() == {
        "prediction": "MALIGNO",
        "confidence": 0.93,
        "top_features": [{"feature": "radius", "value": 2.0, "impact": 1.4}],
    }


def test_diagnose_endpoint_turns_domain_validation_into_400(monkeypatch) -> None:
    def raise_value_error(patient_index: int):
        raise ValueError("bad patient")

    monkeypatch.setattr(api, "diagnose_patient", raise_value_error)

    response = client.post("/diagnose", json={"patient_index": 999})

    assert response.status_code == 400
    assert response.json()["detail"] == "bad patient"


def test_explain_endpoint_serializes_llm_response(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "explain_diagnosis",
        lambda **kwargs: api.ExplainResponse(
            explanation="Explicacao",
            disclaimer="Aviso",
            details={"nivel_confianca": "alto"},
        ),
    )

    response = client.post(
        "/explain",
        json={
            "prediction": "MALIGNO",
            "confidence": 0.9,
            "top_features": [{"feature": "radius", "value": 2.0, "impact": 1.0}],
        },
    )

    assert response.status_code == 200
    assert response.json()["explanation"] == "Explicacao"


def test_chat_endpoint_rejects_domain_validation_error(monkeypatch) -> None:
    def raise_value_error(question: str, context=None):
        raise ValueError("empty question")

    monkeypatch.setattr(api, "chat_about_diagnosis", raise_value_error)

    response = client.post("/chat", json={"question": " "})

    assert response.status_code == 400
    assert response.json()["detail"] == "empty question"


def test_ag_results_endpoint_uses_real_artifact() -> None:
    response = client.get("/ag-results")

    assert response.status_code == 200
    assert response.json()["best_experiment"] == "Exp1_Pequeno"
