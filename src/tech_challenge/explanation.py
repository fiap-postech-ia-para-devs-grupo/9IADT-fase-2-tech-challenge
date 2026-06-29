from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tech_challenge.diagnosis import DiagnosisResult, FeatureImpact
from tech_challenge.llm.medical_agent import DiagnosisInput, MedicalDiagnosisAgent

DISCLAIMER = "Este diagnostico e um apoio computacional. A decisao clinica final e responsabilidade do medico."
FeaturePayloadValue = str | int | float


@dataclass(frozen=True)
class Explanation:
    explanation: str
    disclaimer: str


def explain_diagnosis(
    prediction: str,
    confidence: float,
    top_features: Sequence[FeatureImpact | Mapping[str, FeaturePayloadValue]],
    *,
    api_key: str | None = None,
) -> Explanation:
    normalized_features = [_normalize_feature(feature) for feature in top_features]
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if key:
        return _explain_with_gemini(prediction, confidence, normalized_features, key)

    return _static_explanation(prediction, confidence, normalized_features)


def explain_result(result: DiagnosisResult, *, api_key: str | None = None) -> Explanation:
    return explain_diagnosis(result.prediction, result.confidence, result.top_features, api_key=api_key)


def _explain_with_gemini(
    prediction: str,
    confidence: float,
    top_features: Sequence[FeatureImpact],
    api_key: str,
) -> Explanation:
    agent = MedicalDiagnosisAgent(api_key=api_key)
    response = agent.explain(
        DiagnosisInput(
            features={feature.feature: feature.impact for feature in top_features},
            prediction=1 if prediction.upper() == "MALIGNO" else 0,
            probability=confidence,
            model_name="AG-ML",
        )
    )

    text = response.get("interpretacao") or response.get("explanation") or response.get("resposta") or str(response)
    disclaimer = response.get("disclaimer") or DISCLAIMER
    return Explanation(explanation=str(text), disclaimer=str(disclaimer))


def _static_explanation(prediction: str, confidence: float, top_features: Sequence[FeatureImpact]) -> Explanation:
    feature_names = ", ".join(feature.feature for feature in top_features[:3])
    text = (
        f"O modelo indicou resultado {prediction.upper()} com {confidence:.0%} de confianca. "
        f"As variaveis com maior impacto foram: {feature_names}. "
        "Use esta saida como apoio para correlacao clinica e validacao por profissional de saude."
    )
    return Explanation(explanation=text, disclaimer=DISCLAIMER)


def _normalize_feature(feature: FeatureImpact | Mapping[str, FeaturePayloadValue]) -> FeatureImpact:
    if isinstance(feature, FeatureImpact):
        return feature
    return FeatureImpact(
        feature=str(feature["feature"]),
        value=float(feature["value"]),
        impact=float(feature["impact"]),
    )
