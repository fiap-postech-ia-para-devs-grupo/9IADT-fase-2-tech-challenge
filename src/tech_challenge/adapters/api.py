from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tech_challenge.diagnosis import diagnose_patient, patient_metadata
from tech_challenge.experiments import load_ag_results
from tech_challenge.explanation import (
    LLMConfigurationError,
    LLMProviderError,
    chat_about_diagnosis,
    explain_diagnosis,
)

app = FastAPI(title="Tech Challenge Fase 2 API", version="0.1.0")


class FeatureImpact(BaseModel):
    feature: str
    value: float
    impact: float


class DiagnoseRequest(BaseModel):
    patient_index: int


class DiagnoseResponse(BaseModel):
    prediction: str
    confidence: float
    top_features: list[FeatureImpact]


class ExplainRequest(BaseModel):
    prediction: str
    confidence: float
    top_features: list[FeatureImpact]


class ExplainResponse(BaseModel):
    explanation: str
    disclaimer: str
    details: dict[str, Any]


class ChatRequest(BaseModel):
    question: str
    context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    answer: str
    details: dict[str, Any]


class PatientMetadataResponse(BaseModel):
    count: int
    min_index: int
    max_index: int


class AGExperiment(BaseModel):
    name: str
    population: int
    generations: int
    mutation_rate: float
    best_fitness: float
    test_metrics: dict[str, float]
    convergence: list[float]


class Baseline(BaseModel):
    name: str
    metrics: dict[str, float]


class AGResultsResponse(BaseModel):
    experiments: list[AGExperiment]
    baseline: Baseline
    best_experiment: str
    best_config: dict[str, Any]
    rf_baseline: dict[str, Any]
    best_model: dict[str, Any]
    source: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/patients/metadata", response_model=PatientMetadataResponse)
def patients_metadata() -> PatientMetadataResponse:
    return PatientMetadataResponse(**asdict(patient_metadata()))


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest) -> DiagnoseResponse:
    try:
        result = diagnose_patient(req.patient_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DiagnoseResponse(
        prediction=result.prediction,
        confidence=result.confidence,
        top_features=[FeatureImpact(**asdict(feature)) for feature in result.top_features],
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> ExplainResponse:
    try:
        result = explain_diagnosis(
            prediction=req.prediction,
            confidence=req.confidence,
            top_features=[feature.model_dump() for feature in req.top_features],
        )
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ExplainResponse(explanation=result.explanation, disclaimer=result.disclaimer, details=result.details)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        result = chat_about_diagnosis(question=req.question, context=req.context)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(answer=result.answer, details=result.details)


@app.get("/ag-results", response_model=AGResultsResponse)
def ag_results() -> AGResultsResponse:
    return AGResultsResponse(**asdict(load_ag_results()))
