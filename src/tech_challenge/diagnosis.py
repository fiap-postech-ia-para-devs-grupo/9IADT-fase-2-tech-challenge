from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from tech_challenge.paths import BEST_MODEL, BREAST_CANCER_DATASET


@dataclass(frozen=True)
class FeatureImpact:
    feature: str
    value: float
    impact: float


@dataclass(frozen=True)
class DiagnosisResult:
    prediction: str
    prediction_id: int
    confidence: float
    top_features: list[FeatureImpact]
    features_scaled: dict[str, float]
    genes: dict[str, Any]


@dataclass(frozen=True)
class PatientMetadata:
    count: int
    min_index: int
    max_index: int


class DiagnosisModule:
    """Deep diagnosis module: dataset lookup, model loading, prediction, and feature ranking."""

    def __init__(self, model_path: Path = BEST_MODEL, dataset_path: Path = BREAST_CANCER_DATASET) -> None:
        bundle = joblib.load(model_path)

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.features = list(bundle["features"])
        self.genes = dict(bundle.get("genes", {}))
        self.dataset = pd.read_csv(dataset_path)

    def diagnose_patient(self, patient_index: int) -> DiagnosisResult:
        if patient_index < 0 or patient_index >= len(self.dataset):
            msg = f"patient_index must be between 0 and {len(self.dataset) - 1}"
            raise ValueError(msg)

        raw_features = self._patient_features(patient_index)
        return self.diagnose_features(raw_features)

    def patient_metadata(self) -> PatientMetadata:
        count = len(self.dataset)
        return PatientMetadata(count=count, min_index=0, max_index=count - 1)

    def diagnose_features(self, raw_features: dict[str, float]) -> DiagnosisResult:
        x = pd.DataFrame([raw_features])[self.features]
        x_scaled = self.scaler.transform(x)
        x_scaled_df = pd.DataFrame(x_scaled, columns=pd.Index(self.features))

        prediction_id = int(self.model.predict(x_scaled_df)[0])
        probability = float(self.model.predict_proba(x_scaled_df)[0][prediction_id])
        features_scaled = {feature: float(value) for feature, value in zip(self.features, x_scaled[0])}

        return DiagnosisResult(
            prediction="MALIGNO" if prediction_id == 1 else "BENIGNO",
            prediction_id=prediction_id,
            confidence=probability,
            top_features=self._rank_features(raw_features, features_scaled),
            features_scaled=features_scaled,
            genes=self.genes,
        )

    def _patient_features(self, patient_index: int) -> dict[str, float]:
        row = self.dataset.iloc[patient_index]
        return {feature: float(row[self._dataset_column(feature)]) for feature in self.features}

    def _dataset_column(self, feature: str) -> str:
        candidates: list[str]
        if feature.startswith("mean "):
            base = feature.removeprefix("mean ")
            candidates = [f"{base}_mean", f"{base.replace(' ', '_')}_mean"]
        elif feature.endswith(" error"):
            base = feature.removesuffix(" error")
            candidates = [f"{base}_se", f"{base.replace(' ', '_')}_se"]
        elif feature.startswith("worst "):
            base = feature.removeprefix("worst ")
            candidates = [f"{base}_worst", f"{base.replace(' ', '_')}_worst"]
        else:
            candidates = [feature, feature.replace(" ", "_")]

        for candidate in candidates:
            if candidate in self.dataset.columns:
                return candidate

        msg = f"Could not map model feature {feature!r} to a dataset column"
        raise KeyError(msg)

    def _rank_features(self, raw_features: dict[str, float], features_scaled: dict[str, float]) -> list[FeatureImpact]:
        coefficients = getattr(self.model, "coef_", None)
        if coefficients is not None and len(coefficients) > 0:
            impacts = {
                feature: float(coef) * features_scaled[feature] for feature, coef in zip(self.features, coefficients[0])
            }
        else:
            impacts = features_scaled

        ranked = sorted(self.features, key=lambda feature: abs(impacts[feature]), reverse=True)[:5]
        return [
            FeatureImpact(feature=feature, value=float(raw_features[feature]), impact=float(impacts[feature]))
            for feature in ranked
        ]


@lru_cache(maxsize=1)
def get_diagnosis_module() -> DiagnosisModule:
    return DiagnosisModule()


def diagnose_patient(patient_index: int) -> DiagnosisResult:
    return get_diagnosis_module().diagnose_patient(patient_index)


def patient_metadata() -> PatientMetadata:
    return get_diagnosis_module().patient_metadata()
