from __future__ import annotations

import unittest
from typing import Any, cast

import pandas as pd

from tech_challenge.adapters.api import ag_results, patients_metadata
from tech_challenge.experiments import load_ag_results
from tech_challenge.explanation import chat_about_diagnosis
from tech_challenge.llm.medical_agent import MedicalDiagnosisAgent
from tech_challenge.presentation.formatting import (
    ag_result_rows,
    chat_answer_text,
    filter_patient_table,
    paginate_patient_table,
    patient_table_rows,
    sort_patient_table,
)


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
        agent.client = cast(Any, _FakeClient('```json\n{"resposta": "Texto de chat."}\n```'))

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
        self.assertEqual(chat_answer_text('{"resposta": "Texto de chat."}'), "Texto de chat.")


class AGResultRowsTests(unittest.TestCase):
    def test_ag_table_columns_are_arrow_friendly_strings(self) -> None:
        rows = ag_result_rows(
            [
                {
                    "name": "Exp",
                    "best_fitness": 0.95,
                    "population": 20,
                    "generations": 5,
                    "mutation_rate": 0.1,
                }
            ],
            baseline={"name": "Baseline", "cv_fitness": 0.9},
        )

        self.assertEqual([type(row["Pop"]) for row in rows], [str, str])
        self.assertEqual([type(row["Gen"]) for row in rows], [str, str])
        self.assertEqual([type(row["Mut"]) for row in rows], [str, str])


class AGResultsTests(unittest.TestCase):
    def test_ag_results_are_loaded_from_artifact_with_real_best_config_shape(self) -> None:
        results = load_ag_results()

        best = max(results.experiments, key=lambda experiment: experiment.best_fitness)

        self.assertEqual(results.schema_version, 2)
        self.assertEqual(results.best_experiment, best.name)
        self.assertNotIn("model_type", results.best_config)
        self.assertEqual(results.baseline.name, "RF_Baseline")
        self.assertEqual(len(results.experiments[0].history.best), results.experiments[0].generations)
        self.assertEqual(len(results.experiments[0].history.mean), results.experiments[0].generations)

    def test_api_ag_results_exposes_expanded_artifact_shape(self) -> None:
        response = ag_results()

        payload = response.model_dump()
        self.assertEqual(payload["schema_version"], 2)
        self.assertIn("baseline", payload)
        self.assertIn("history", payload["experiments"][0])
        self.assertIn("final_comparison", payload)


class PatientMetadataTests(unittest.TestCase):
    def test_api_patient_metadata_comes_from_dataset(self) -> None:
        response = patients_metadata()

        self.assertEqual(response.model_dump(), {"count": 569, "min_index": 0, "max_index": 568})


class PatientTableTests(unittest.TestCase):
    def test_patient_table_rows_exposes_main_columns_with_dataset_index(self) -> None:
        dataset = pd.DataFrame(
            [
                {
                    "id": 10,
                    "diagnosis": "M",
                    "radius_mean": 1.1,
                    "texture_mean": 2.2,
                    "perimeter_mean": 3.3,
                    "area_mean": 4.4,
                    "concavity_mean": 5.5,
                },
                {
                    "id": 20,
                    "diagnosis": "B",
                    "radius_mean": 6.6,
                    "texture_mean": 7.7,
                    "perimeter_mean": 8.8,
                    "area_mean": 9.9,
                    "concavity_mean": 10.1,
                },
            ]
        )

        rows = patient_table_rows(dataset, min_index=1, max_index=1)

        self.assertEqual(rows.to_dict("records")[0]["Índice"], 1)
        self.assertEqual(rows.to_dict("records")[0]["Diagnóstico real"], "Benigno")
        self.assertEqual(
            rows.columns.tolist(),
            [
                "Índice",
                "ID",
                "Diagnóstico real",
                "Raio médio",
                "Textura média",
                "Perímetro médio",
                "Área média",
                "Concavidade média",
            ],
        )

    def test_patient_table_searches_only_by_id(self) -> None:
        rows = pd.DataFrame(
            [
                {"Índice": 0, "ID": 10, "Diagnóstico real": "Maligno", "Raio médio": 3.0},
                {"Índice": 1, "ID": 20, "Diagnóstico real": "Benigno", "Raio médio": 1.0},
                {"Índice": 2, "ID": 30, "Diagnóstico real": "Benigno", "Raio médio": 2.0},
            ]
        )

        filtered_by_diagnosis = filter_patient_table(rows, "benigno")
        filtered_by_id = filter_patient_table(rows, "20")

        self.assertTrue(filtered_by_diagnosis.empty)
        self.assertEqual(filtered_by_id["Índice"].tolist(), [1])

    def test_patient_table_filters_by_real_diagnosis_label(self) -> None:
        rows = pd.DataFrame(
            [
                {"Índice": 0, "ID": 10, "Diagnóstico real": "Maligno", "Raio médio": 3.0},
                {"Índice": 1, "ID": 20, "Diagnóstico real": "Benigno", "Raio médio": 1.0},
                {"Índice": 2, "ID": 30, "Diagnóstico real": "Benigno", "Raio médio": 2.0},
            ]
        )

        filtered_rows = filter_patient_table(rows, "", diagnosis_label="Benigno")

        self.assertEqual(filtered_rows["Índice"].tolist(), [1, 2])

    def test_patient_table_sort_and_paginate(self) -> None:
        rows = pd.DataFrame(
            [
                {"Índice": 0, "ID": 10, "Diagnóstico real": "Maligno", "Raio médio": 3.0},
                {"Índice": 1, "ID": 20, "Diagnóstico real": "Benigno", "Raio médio": 1.0},
                {"Índice": 2, "ID": 30, "Diagnóstico real": "Benigno", "Raio médio": 2.0},
            ]
        )

        sorted_rows = sort_patient_table(rows, "Raio médio", ascending=False)
        page = paginate_patient_table(sorted_rows, page=1, page_size=1)

        self.assertEqual(sorted_rows["Índice"].tolist(), [0, 2, 1])
        self.assertEqual(page["Índice"].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
