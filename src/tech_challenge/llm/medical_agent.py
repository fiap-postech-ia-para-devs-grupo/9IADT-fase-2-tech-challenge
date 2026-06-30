"""
Agente responsável por interpretar saídas de modelos de ML
usando LLM (Gemini).

NÃO realiza classificação.
Apenas gera interpretação clínica estruturada.

Projeto: Tech Challenge - Fase 2
"""

import json
from dataclasses import dataclass

from google import genai
from google.genai import types

from tech_challenge.llm.prompts import (
    CHAT_PROMPT,
    MEDICAL_PROMPT_V3,
    SYSTEM_INSTRUCTION,
)


@dataclass
class DiagnosisInput:
    """
    Estrutura de entrada do agente LLM
    """

    features: dict
    prediction: int
    probability: float
    model_name: str = "AG-ML"


class MedicalDiagnosisAgent:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):

        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def _build_prompt(self, inp: DiagnosisInput) -> str:

        prediction = "MALIGNO" if inp.prediction == 1 else "BENIGNO"
        confidence = inp.probability * 100

        top_features = sorted(inp.features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        features_txt = "\n".join([f"- {k}: {v:.4f}" for k, v in top_features])

        prompt = f"""
{MEDICAL_PROMPT_V3}

============================================================
CASO CLÍNICO
============================================================

Modelo: {inp.model_name}
Predição: {prediction}
Probabilidade: {confidence:.2f} %

Top Features:
{features_txt}

============================================================
Retorne SOMENTE JSON válido.
"""

        return prompt

    def _clean_json(self, text: str) -> str:

        if not text:
            return text

        text = text.strip()

        # remove markdown
        if "```" in text:
            text = text.replace("```json", "")
            text = text.replace("```", "")

        text = text.strip()

        # tenta extrair apenas JSON válido
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:
            text = text[start : end + 1]

        return text

    def explain(self, inp: DiagnosisInput) -> dict:

        prompt = self._build_prompt(inp)

        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
            contents=prompt,
        )

        response_text = response.text or ""
        cleaned = self._clean_json(response_text)

        try:
            return json.loads(cleaned)

        except Exception:
            return {"error": "json_invalid", "raw": cleaned}

    def chat(self, question: str, context: dict | None = None) -> dict:

        ctx = ""
        if context:
            ctx = f"\nCONTEXTO:\n{json.dumps(context, indent=2, ensure_ascii=False)}"

        prompt = f"""
{CHAT_PROMPT}

Pergunta do médico:
{question}

{ctx}
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        response_text = response.text or ""
        cleaned = self._clean_json(response_text)

        try:
            return json.loads(cleaned)

        except Exception:
            return {"resposta": cleaned}


def gerar_explicacao(predicao: int, confianca: float, top_features: dict) -> dict:
    """
    Interface simples exigida para integração externa (Streamlit/API)
    """

    return {
        "predicao": "maligno" if predicao == 1 else "benigno",
        "confianca": float(confianca),
        "top_features": [{"nome": k, "valor": float(v)} for k, v in top_features.items()],
    }
