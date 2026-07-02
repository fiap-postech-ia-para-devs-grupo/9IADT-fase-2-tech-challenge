from __future__ import annotations

import numpy as np
import pandas as pd

from tech_challenge.diagnosis_pipeline import DiagnosisPipeline


class FakeScaler:
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.to_numpy(dtype=float) / 2


class FakeModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([0])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([[0.7, 0.3]])


class FakeAgent:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def explain(self, inp) -> dict:
        return {
            "interpretacao": f"{inp.model_name}:{inp.prediction}:{inp.probability}",
            "features": inp.features,
        }


class FailingAgent(FakeAgent):
    def explain(self, inp) -> dict:
        raise RuntimeError("provider down")


def _patch_bundle(monkeypatch) -> None:
    monkeypatch.setattr(
        "tech_challenge.diagnosis_pipeline.joblib.load",
        lambda _path: {
            "model": FakeModel(),
            "scaler": FakeScaler(),
            "features": ["radius", "texture"],
            "genes": {"model_type": "logreg"},
        },
    )


def test_pipeline_predict_returns_ml_output_and_llm_explanation(monkeypatch) -> None:
    _patch_bundle(monkeypatch)
    monkeypatch.setattr("tech_challenge.diagnosis_pipeline.MedicalDiagnosisAgent", FakeAgent)

    result = DiagnosisPipeline("model.pkl", "key").predict({"radius": 4.0, "texture": 2.0})

    assert result["prediction"] == "benigno"
    assert result["prediction_id"] == 0
    assert result["probability"] == 0.7
    assert result["features_scaled"] == {"radius": 2.0, "texture": 1.0}
    assert result["explanation"]["interpretacao"] == "AG-logreg:0:0.7"


def test_pipeline_predict_wraps_llm_failures(monkeypatch) -> None:
    _patch_bundle(monkeypatch)
    monkeypatch.setattr("tech_challenge.diagnosis_pipeline.MedicalDiagnosisAgent", FailingAgent)

    result = DiagnosisPipeline("model.pkl", "key").predict({"radius": 4.0, "texture": 2.0})

    assert result["explanation"] == {"error": "llm_failure", "message": "provider down"}


def test_pipeline_predict_raw_skips_llm(monkeypatch) -> None:
    _patch_bundle(monkeypatch)
    monkeypatch.setattr("tech_challenge.diagnosis_pipeline.MedicalDiagnosisAgent", FakeAgent)

    result = DiagnosisPipeline("model.pkl", "key").predict_raw({"radius": 4.0, "texture": 2.0})

    assert result == {"prediction": 0, "probability": 0.7}
