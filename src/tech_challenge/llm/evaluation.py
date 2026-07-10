from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tech_challenge.paths import RESULTS_DIR

EVALUATIONS_PATH = RESULTS_DIR / "llm_evaluations.jsonl"


@dataclass(frozen=True)
class AutomaticChecks:
    structured_response: bool
    disclaimer_present: bool
    prediction_consistent: bool
    shap_features_referenced: bool


def automatic_checks(
    prediction: str,
    top_features: list[dict[str, Any]],
    explanation: str,
    disclaimer: str,
    details: dict[str, Any],
) -> AutomaticChecks:
    serialized_details = json.dumps(details, ensure_ascii=False).lower()
    combined_text = f"{explanation} {serialized_details}".lower()
    expected_term = "malign" if prediction.upper() == "MALIGNO" else "benign"
    referenced_features = sum(
        1 for feature in top_features if str(feature.get("feature", "")).lower() in serialized_details
    )
    return AutomaticChecks(
        structured_response=bool(details) and "error" not in details,
        disclaimer_present=bool(disclaimer.strip()),
        prediction_consistent=expected_term in combined_text,
        shap_features_referenced=referenced_features > 0,
    )


def save_evaluation(
    *,
    prediction: str,
    checks: AutomaticChecks,
    ratings: dict[str, int],
    comment: str,
    path: Path = EVALUATIONS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now(UTC).isoformat(),
        "prediction": prediction,
        "automatic_checks": asdict(checks),
        "ratings": ratings,
        "comment": comment.strip(),
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluation_summary(path: Path = EVALUATIONS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"count": 0, "averages": {}}

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rating_names = ("clareza", "coerencia", "seguranca", "utilidade")
    averages = {name: sum(float(record["ratings"][name]) for record in records) / len(records) for name in rating_names}
    return {"count": len(records), "averages": averages}
