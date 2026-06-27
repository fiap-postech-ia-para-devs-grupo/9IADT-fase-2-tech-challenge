"""
Pipeline de inferência do modelo + integração com LLM (Gemini)

Fluxo:
ML Model → Previsão → Features normalizadas → LLM → Explicação

Projeto: Tech Challenge - Fase 2
"""

import joblib
import pandas as pd

from llm.medical_agent import MedicalDiagnosisAgent, DiagnosisInput

class DiagnosisPipeline:

    def __init__(self, model_path: str, gemini_api_key: str):

        bundle = joblib.load(model_path)

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.features = bundle["features"]
        self.genes = bundle["genes"]

        self.agent = MedicalDiagnosisAgent(api_key=gemini_api_key)

    def predict(self, raw_features: dict) -> dict:

        X = pd.DataFrame([raw_features])[self.features]

        X_sc = self.scaler.transform(X)

        pred = int(self.model.predict(X_sc)[0])

        proba_vec = self.model.predict_proba(X_sc)[0]
        proba = float(proba_vec[pred])

        feats_sc = dict(zip(self.features, X_sc[0]))

        inp = DiagnosisInput(
            features=feats_sc,
            prediction=pred,
            probability=proba,
            model_name=f"AG-{self.genes.get('model_type','unknown')}"
        )

        try:
            explanation = self.agent.explain(inp)

        except Exception as e:
            explanation = {
                "error": "llm_failure",
                "message": str(e)
            }

        return {
            "prediction": "maligno" if pred == 1 else "benigno",
            "prediction_id": pred,
            "probability": proba,
            "genes": self.genes,
            "features_scaled": feats_sc,
            "explanation": explanation
        }


    def predict_raw(self, raw_features: dict) -> dict:
        """
        Retorna apenas ML sem LLM (útil para debug)
        """

        X = pd.DataFrame([raw_features])[self.features]
        X_sc = self.scaler.transform(X)

        pred = int(self.model.predict(X_sc)[0])
        proba = float(self.model.predict_proba(X_sc)[0][pred])

        return {
            "prediction": pred,
            "probability": proba
        }