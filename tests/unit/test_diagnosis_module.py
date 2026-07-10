from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tech_challenge.diagnosis import DiagnosisModule


class FakeScaler:
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.to_numpy(dtype=float) / 10


class FakeModel:
    coef_ = np.array([[2.0, -3.0, 0.5]])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.to_numpy(dtype=float)
        first_feature = values[0, 0]
        return np.array([1 if first_feature > 0 else 0])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([[0.2, 0.8]])


class FakeTreeModel:
    feature_importances_ = np.array([0.3, 0.7])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([1])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([[0.1, 0.9]])


def test_diagnose_patient_maps_dataset_columns_and_ranks_impacts(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "patients.csv"
    dataset_path.write_text(
        "id,diagnosis,radius_mean,texture_se,area_worst\n10,M,20.0,5.0,100.0\n",
        encoding="utf-8",
    )

    def fake_load(_path):
        return {
            "model": FakeModel(),
            "scaler": FakeScaler(),
            "features": ["mean radius", "texture error", "worst area"],
            "genes": {"model_type": "logreg"},
        }

    monkeypatch.setattr("tech_challenge.diagnosis.joblib.load", fake_load)

    module = DiagnosisModule(model_path=tmp_path / "model.pkl", dataset_path=dataset_path)
    result = module.diagnose_patient(0)

    assert result.prediction == "MALIGNO"
    assert result.prediction_id == 1
    assert result.confidence == pytest.approx(0.8)
    assert result.features_scaled == {
        "mean radius": 2.0,
        "texture error": 0.5,
        "worst area": 10.0,
    }
    assert [feature.feature for feature in result.top_features] == [
        "worst area",
        "mean radius",
        "texture error",
    ]
    assert result.genes == {"model_type": "logreg"}


def test_diagnose_patient_rejects_out_of_range_index(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "patients.csv"
    dataset_path.write_text("radius_mean\n1.0\n", encoding="utf-8")
    monkeypatch.setattr(
        "tech_challenge.diagnosis.joblib.load",
        lambda _path: {"model": FakeModel(), "scaler": FakeScaler(), "features": ["mean radius"]},
    )

    module = DiagnosisModule(model_path=tmp_path / "model.pkl", dataset_path=dataset_path)

    with pytest.raises(ValueError, match="patient_index must be between 0 and 0"):
        module.diagnose_patient(1)


def test_unknown_feature_column_raises_clear_key_error(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "patients.csv"
    dataset_path.write_text("radius_mean\n1.0\n", encoding="utf-8")
    monkeypatch.setattr(
        "tech_challenge.diagnosis.joblib.load",
        lambda _path: {"model": FakeModel(), "scaler": FakeScaler(), "features": ["mean smoothness"]},
    )

    module = DiagnosisModule(model_path=tmp_path / "model.pkl", dataset_path=dataset_path)

    with pytest.raises(KeyError, match="Could not map model feature"):
        module.diagnose_patient(0)


def test_patient_metadata_uses_dataset_bounds(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "patients.csv"
    dataset_path.write_text("radius_mean\n1.0\n2.0\n", encoding="utf-8")
    monkeypatch.setattr(
        "tech_challenge.diagnosis.joblib.load",
        lambda _path: {"model": FakeModel(), "scaler": FakeScaler(), "features": ["mean radius"]},
    )

    module = DiagnosisModule(model_path=tmp_path / "model.pkl", dataset_path=dataset_path)

    assert module.patient_metadata().count == 2
    assert module.patient_metadata().min_index == 0
    assert module.patient_metadata().max_index == 1


def test_tree_model_ranks_local_shap_impacts(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "patients.csv"
    dataset_path.write_text("radius_mean,texture_mean\n20.0,5.0\n", encoding="utf-8")
    monkeypatch.setattr(
        "tech_challenge.diagnosis.joblib.load",
        lambda _path: {
            "model": FakeTreeModel(),
            "scaler": FakeScaler(),
            "features": ["mean radius", "mean texture"],
            "genes": {"n_estimators": 100},
        },
    )

    module = DiagnosisModule(model_path=tmp_path / "model.pkl", dataset_path=dataset_path)
    monkeypatch.setattr(module, "_tree_shap_values", lambda frame, prediction_id: [-0.2, 1.4])

    result = module.diagnose_patient(0)

    assert [feature.feature for feature in result.top_features] == ["mean texture", "mean radius"]
    assert [feature.impact for feature in result.top_features] == [1.4, -0.2]
