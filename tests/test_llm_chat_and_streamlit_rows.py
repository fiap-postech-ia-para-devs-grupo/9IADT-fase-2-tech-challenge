from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from tech_challenge.adapters.api import app
from tech_challenge.experiments import load_ag_results
from tech_challenge.explanation import chat_about_diagnosis
from tech_challenge.llm.medical_agent import MedicalDiagnosisAgent
from tech_challenge.presentation.streamlit_app import _ag_result_rows, _chat_answer_text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, text: str) -> None:
        self.text = text

    def generate_content(self, **_: object) -> _FakeResponse:
        return _FakeResponse(self.text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.models = _FakeModels(text)


class ChatParsingTests(unittest.TestCase):
    def test_agent_chat_extracts_fenced_json_answer(self) -> None:
        agent = MedicalDiagnosisAgent.__new__(MedicalDiagnosisAgent)
        agent.model_name = "fake"
        agent.client = _FakeClient('```json\n{"resposta": "Texto de chat."}\n```')

        self.assertEqual(agent.chat("Pergunta?")["resposta"], "Texto de chat.")

    def test_chat_about_diagnosis_never_returns_raw_json_when_answer_is_json_text(self) -> None:
        original_chat = MedicalDiagnosisAgent.chat

        def fake_chat(self: MedicalDiagnosisAgent, question: str, context: dict | None = None) -> dict:
            return {"resposta": '{"resposta": "Texto de chat."}'}

        MedicalDiagnosisAgent.chat = fake_chat
        try:
            result = chat_about_diagnosis("Pergunta?", api_key="fake")
        finally:
            MedicalDiagnosisAgent.chat = original_chat

        self.assertEqual(result.answer, "Texto de chat.")

    def test_streamlit_chat_answer_text_extracts_json_string(self) -> None:
        self.assertEqual(_chat_answer_text('{"resposta": "Texto de chat."}'), "Texto de chat.")


class AGResultRowsTests(unittest.TestCase):
    def test_ag_table_columns_are_arrow_friendly_strings(self) -> None:
        rows = _ag_result_rows(
            [
                {
                    "name": "Exp",
                    "test_metrics": {"f1": 0.95, "recall": 0.92, "equity_gap": 0.03},
                    "population": 20,
                    "generations": 5,
                    "mutation_rate": 0.1,
                }
            ],
            baseline={"name": "Baseline", "metrics": {"f1": 0.9, "recall": 0.88, "equity_gap": 0.04}},
        )

        self.assertEqual([type(row["Pop"]) for row in rows], [str, str])
        self.assertEqual([type(row["Gen"]) for row in rows], [str, str])
        self.assertEqual([type(row["Mut"]) for row in rows], [str, str])


class AGResultsTests(unittest.TestCase):
    def test_ag_results_are_loaded_from_artifact_with_real_best_config_shape(self) -> None:
        results = load_ag_results()

        self.assertEqual(results.best_experiment, "Exp1_Pequeno")
        self.assertEqual(results.best_config["model_type"], "logreg")
        self.assertEqual(results.best_config["penalty"], "l2")
        self.assertAlmostEqual(results.baseline.metrics["f1"], 0.9512)
        self.assertAlmostEqual(results.experiments[0].test_metrics["f1"], 0.9756)

    def test_api_ag_results_exposes_expanded_artifact_shape(self) -> None:
        client = TestClient(app)

        response = client.get("/ag-results")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["best_config"]["model_type"], "logreg")
        self.assertIn("baseline", payload)
        self.assertIn("test_metrics", payload["experiments"][0])


class PatientMetadataTests(unittest.TestCase):
    def test_api_patient_metadata_comes_from_dataset(self) -> None:
        client = TestClient(app)

        response = client.get("/patients/metadata")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"count": 569, "min_index": 0, "max_index": 568})


if __name__ == "__main__":
    unittest.main()
