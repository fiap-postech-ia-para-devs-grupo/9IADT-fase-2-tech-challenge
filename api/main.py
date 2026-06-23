from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Tech Challenge Fase 2 API", version="0.1.0")


# --- Schemas ---

class DiagnoseRequest(BaseModel):
    patient_index: int


class DiagnoseResponse(BaseModel):
    prediction: str  # "MALIGNO" | "BENIGNO"
    confidence: float
    top_features: list[dict]  # [{"feature": str, "value": float, "impact": float}]


class ExplainRequest(BaseModel):
    prediction: str
    confidence: float
    top_features: list[dict]


class ExplainResponse(BaseModel):
    explanation: str
    disclaimer: str


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest):
    # TODO: load rf_otimizado.pkl, run prediction + SHAP
    return DiagnoseResponse(
        prediction="MALIGNO",
        confidence=0.87,
        top_features=[
            {"feature": "worst concave points", "value": 0.265, "impact": 0.42},
            {"feature": "worst perimeter", "value": 184.6, "impact": 0.31},
            {"feature": "mean concave points", "value": 0.147, "impact": 0.18},
        ],
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    # TODO: call src/llm/agente.py gerar_explicacao()
    return ExplainResponse(
        explanation=(
            "O modelo RandomForest otimizado indicou diagnóstico MALIGNO com 87% de confiança. "
            "As características 'worst concave points' e 'worst perimeter' foram as mais determinantes "
            "para este resultado, sugerindo irregularidades morfológicas na massa analisada. "
            "Recomenda-se encaminhamento para confirmação histopatológica e avaliação clínica completa."
        ),
        disclaimer="Este diagnóstico é um apoio computacional. A decisão clínica final é responsabilidade do médico.",
    )


@app.get("/ag-results")
def ag_results():
    # TODO: load results/ag_results.json
    return {
        "experiments": [
            {"name": "Exp 1", "population": 10, "generations": 30, "mutation_rate": 0.01, "best_f1": 0.951, "convergence": [0.88, 0.91, 0.93, 0.94, 0.951]},
            {"name": "Exp 2", "population": 30, "generations": 50, "mutation_rate": 0.01, "best_f1": 0.967, "convergence": [0.89, 0.93, 0.95, 0.961, 0.967]},
            {"name": "Exp 3", "population": 30, "generations": 50, "mutation_rate": 0.10, "best_f1": 0.943, "convergence": [0.87, 0.90, 0.92, 0.935, 0.943]},
        ],
        "baseline_f1": 0.934,
        "best_config": {
            "experiment": "Exp 2",
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_split": 2,
            "max_features": "sqrt",
        },
    }
