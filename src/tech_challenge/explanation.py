from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from tech_challenge.diagnosis import DiagnosisResult, FeatureImpact
from tech_challenge.llm.medical_agent import DiagnosisInput, MedicalDiagnosisAgent

DISCLAIMER = "Este diagnostico e um apoio computacional. A decisao clinica final e responsabilidade do medico."
FeaturePayloadValue = str | int | float


@dataclass(frozen=True)
class Explanation:
    explanation: str
    disclaimer: str
    details: dict[str, Any]


@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    details: dict[str, Any]


class LLMConfigurationError(RuntimeError):
    pass


class LLMProviderError(RuntimeError):
    pass


def explain_diagnosis(
    prediction: str,
    confidence: float,
    top_features: Sequence[FeatureImpact | Mapping[str, FeaturePayloadValue]],
    *,
    api_key: str | None = None,
) -> Explanation:
    normalized_features = [_normalize_feature(feature) for feature in top_features]
    key = api_key or _google_api_key()
    return _explain_with_gemini(prediction, confidence, normalized_features, key)


def chat_about_diagnosis(
    question: str,
    context: Mapping[str, Any] | None = None,
    *,
    api_key: str | None = None,
) -> ChatAnswer:
    if not question.strip():
        msg = "question must not be empty"
        raise ValueError(msg)

    key = api_key or _google_api_key()
    agent = MedicalDiagnosisAgent(api_key=key)

    try:
        response = agent.chat(question=question, context=dict(context or {}))
    except Exception as exc:
        msg = _provider_error_message("LLM provider failed while generating chat answer", exc)
        raise LLMProviderError(msg) from exc

    if response.get("error"):
        msg = f"LLM provider returned an error: {response['error']}"
        raise LLMProviderError(msg)

    answer = _extract_answer(response)
    if not answer:
        msg = "LLM response did not include an answer"
        raise LLMProviderError(msg)

    return ChatAnswer(answer=str(answer), details=response)


def explain_result(result: DiagnosisResult, *, api_key: str | None = None) -> Explanation:
    return explain_diagnosis(result.prediction, result.confidence, result.top_features, api_key=api_key)


def _explain_with_gemini(
    prediction: str,
    confidence: float,
    top_features: Sequence[FeatureImpact],
    api_key: str,
) -> Explanation:
    agent = MedicalDiagnosisAgent(api_key=api_key)
    try:
        response = agent.explain(
            DiagnosisInput(
                features={feature.feature: feature.impact for feature in top_features},
                prediction=1 if prediction.upper() == "MALIGNO" else 0,
                probability=confidence,
                model_name="AG-ML",
            )
        )
    except Exception as exc:
        msg = _provider_error_message("LLM provider failed while generating explanation", exc)
        raise LLMProviderError(msg) from exc

    if response.get("error"):
        msg = f"LLM provider returned an error: {response['error']}"
        raise LLMProviderError(msg)

    text = response.get("interpretacao") or response.get("explanation") or response.get("resposta") or str(response)
    disclaimer = response.get("disclaimer") or DISCLAIMER
    return Explanation(explanation=str(text), disclaimer=str(disclaimer), details=response)


def _normalize_feature(feature: FeatureImpact | Mapping[str, FeaturePayloadValue]) -> FeatureImpact:
    if isinstance(feature, FeatureImpact):
        return feature
    return FeatureImpact(
        feature=str(feature["feature"]),
        value=float(feature["value"]),
        impact=float(feature["impact"]),
    )


def _extract_answer(response: Mapping[str, Any]) -> str | None:
    answer = response.get("resposta") or response.get("answer") or response.get("response")
    if not isinstance(answer, str):
        return str(answer) if answer is not None else None

    text = answer.strip()
    if not text.startswith("{"):
        return text

    try:
        payload = json.loads(text)
    except ValueError:
        return text

    if not isinstance(payload, Mapping):
        return text

    nested_answer = payload.get("resposta") or payload.get("answer") or payload.get("response")
    return str(nested_answer) if nested_answer is not None else text


def _google_api_key() -> str:
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        msg = "GOOGLE_API_KEY is required for LLM explanation. Set it in your environment or .env file."
        raise LLMConfigurationError(msg)
    return key


def _provider_error_message(prefix: str, exc: Exception) -> str:
    detail = str(exc).strip()
    if not detail:
        return prefix
    return f"{prefix}: {detail[:500]}"
