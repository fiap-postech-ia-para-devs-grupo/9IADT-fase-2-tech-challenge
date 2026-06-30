from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tech_challenge.paths import AG_EXPERIMENT_RESULTS


@dataclass(frozen=True)
class AGExperiment:
    name: str
    population: int
    generations: int
    mutation_rate: float
    best_fitness: float
    test_metrics: dict[str, float]
    convergence: list[float]


@dataclass(frozen=True)
class Baseline:
    name: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class AGResults:
    experiments: list[AGExperiment]
    baseline: Baseline
    best_experiment: str
    best_config: dict[str, Any]
    rf_baseline: dict[str, Any]
    best_model: dict[str, Any]
    source: dict[str, Any]


def load_ag_results(path: Path = AG_EXPERIMENT_RESULTS) -> AGResults:
    """Load presentation-ready AG results exported from the study notebook."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return AGResults(
        experiments=[
            AGExperiment(
                name=str(experiment["name"]),
                population=int(experiment["population"]),
                generations=int(experiment["generations"]),
                mutation_rate=float(experiment["mutation_rate"]),
                best_fitness=float(experiment["best_fitness"]),
                test_metrics={key: float(value) for key, value in experiment["test_metrics"].items()},
                convergence=[float(value) for value in experiment["convergence"]],
            )
            for experiment in payload["experiments"]
        ],
        baseline=Baseline(
            name=str(payload["baseline"]["name"]),
            metrics={key: float(value) for key, value in payload["baseline"]["metrics"].items()},
        ),
        best_experiment=str(payload["best_experiment"]),
        best_config=dict(payload["best_config"]),
        rf_baseline=dict(payload.get("rf_baseline", {})),
        best_model=dict(payload.get("best_model", {})),
        source=dict(payload.get("source", {})),
    )
