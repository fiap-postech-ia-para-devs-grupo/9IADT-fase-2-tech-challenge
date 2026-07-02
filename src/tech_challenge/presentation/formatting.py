from __future__ import annotations

import json
from typing import Any, cast

import pandas as pd

PATIENT_TABLE_COLUMNS = {
    "patient_index": "Índice",
    "id": "ID",
    "diagnosis_label": "Diagnóstico real",
    "radius_mean": "Raio médio",
    "texture_mean": "Textura média",
    "perimeter_mean": "Perímetro médio",
    "area_mean": "Área média",
    "concavity_mean": "Concavidade média",
}


def percentage_text(value: float) -> str:
    percentage = value * 100
    text = f"{percentage:.2f}".rstrip("0").rstrip(".")
    return f"{text}%"


def chat_answer_text(answer: Any) -> str:
    if isinstance(answer, dict):
        return str(answer.get("resposta") or answer.get("answer") or answer.get("response") or answer)

    text = str(answer).strip()
    if not text.startswith("{"):
        return text

    try:
        payload = json.loads(text)
    except ValueError:
        return text

    if isinstance(payload, dict):
        return str(payload.get("resposta") or payload.get("answer") or payload.get("response") or text)
    return text


def ag_result_rows(experiments: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "Modelo": baseline["name"],
            "F1": baseline["metrics"]["f1"],
            "Recall": baseline["metrics"].get("recall"),
            "Equity Gap": baseline["metrics"].get("equity_gap"),
            "Pop": "-",
            "Gen": "-",
            "Mut": "-",
        }
    ]
    for experiment in experiments:
        metrics = experiment["test_metrics"]
        rows.append(
            {
                "Modelo": experiment["name"],
                "F1": metrics["f1"],
                "Recall": metrics.get("recall"),
                "Equity Gap": metrics.get("equity_gap"),
                "Pop": str(experiment["population"]),
                "Gen": str(experiment["generations"]),
                "Mut": f"{experiment['mutation_rate']:.0%}",
            }
        )
    return rows


def patient_table_rows(dataset: pd.DataFrame, min_index: int, max_index: int) -> pd.DataFrame:
    rows = dataset.reset_index(names="patient_index")
    rows = rows[(rows["patient_index"] >= min_index) & (rows["patient_index"] <= max_index)].copy()
    diagnosis = cast(pd.Series, rows["diagnosis"])
    rows["diagnosis_label"] = diagnosis.map({"M": "Maligno", "B": "Benigno"}).fillna(diagnosis)
    return rows.loc[:, list(PATIENT_TABLE_COLUMNS)].rename(columns=PATIENT_TABLE_COLUMNS)


def filter_patient_table(rows: pd.DataFrame, query: str, diagnosis_label: str = "Todos") -> pd.DataFrame:
    normalized_query = query.strip()
    filtered_rows = rows

    if not normalized_query:
        filtered_rows = rows
    else:
        id_matches = cast(pd.Series, rows["ID"].astype(str).str.contains(normalized_query, na=False))
        filtered_rows = rows[id_matches]

    if diagnosis_label != "Todos":
        diagnosis_matches = cast(pd.Series, filtered_rows["Diagnóstico real"] == diagnosis_label)
        filtered_rows = filtered_rows[diagnosis_matches]

    return cast(pd.DataFrame, filtered_rows)


def sort_patient_table(rows: pd.DataFrame, sort_column: str, ascending: bool) -> pd.DataFrame:
    return cast(pd.DataFrame, rows.sort_values(sort_column, ascending=ascending, kind="mergesort"))


def paginate_patient_table(rows: pd.DataFrame, page: int, page_size: int) -> pd.DataFrame:
    start = (page - 1) * page_size
    end = start + page_size
    return cast(pd.DataFrame, rows.iloc[start:end])
