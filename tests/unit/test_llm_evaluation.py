from __future__ import annotations

from tech_challenge.llm.evaluation import automatic_checks, evaluation_summary, save_evaluation


def test_automatic_checks_validate_structure_prediction_and_shap_references() -> None:
    checks = automatic_checks(
        prediction="MALIGNO",
        top_features=[{"feature": "worst radius", "impact": 1.2}],
        explanation="O modelo encontrou um padrão maligno.",
        disclaimer="Isto não substitui avaliação médica.",
        details={"features_relevantes": [{"nome": "worst radius"}]},
    )

    assert checks.structured_response
    assert checks.disclaimer_present
    assert checks.prediction_consistent
    assert checks.shap_features_referenced


def test_evaluations_are_persisted_without_patient_features(tmp_path) -> None:
    path = tmp_path / "evaluations.jsonl"
    checks = automatic_checks("BENIGNO", [], "Achado benigno", "Aviso", {"interpretacao": "benigno"})
    ratings = {"clareza": 5, "coerencia": 4, "seguranca": 5, "utilidade": 4}

    save_evaluation(prediction="BENIGNO", checks=checks, ratings=ratings, comment="Adequada", path=path)
    summary = evaluation_summary(path)

    assert summary == {
        "count": 1,
        "averages": {"clareza": 5.0, "coerencia": 4.0, "seguranca": 5.0, "utilidade": 4.0},
    }
    assert "patient" not in path.read_text(encoding="utf-8").lower()
