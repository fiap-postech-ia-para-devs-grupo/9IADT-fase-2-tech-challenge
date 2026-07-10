from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tech_challenge.paths import AG_EXPERIMENT_RESULTS


@dataclass(frozen=True)
class FitnessDefinition:
    formula: str
    cv_folds: int
    seed: int


@dataclass(frozen=True)
class AGHistory:
    generation: list[int]
    best: list[float]
    mean: list[float]


@dataclass(frozen=True)
class AGExperiment:
    name: str
    population: int
    generations: int
    mutation_rate: float
    best_fitness: float
    best_config: dict[str, Any]
    history: AGHistory


@dataclass(frozen=True)
class Baseline:
    name: str
    cv_fitness: float
    cv_metrics: dict[str, float]
    test_metrics: dict[str, float]
    false_negatives: int


@dataclass(frozen=True)
class AGResults:
    schema_version: int
    fitness: FitnessDefinition
    experiments: list[AGExperiment]
    baseline: Baseline
    best_experiment: str
    best_config: dict[str, Any]
    final_comparison: dict[str, Any]
    subgroup_analysis: dict[str, Any]
    source: dict[str, Any]


def load_ag_results(path: Path = AG_EXPERIMENT_RESULTS) -> AGResults:
    """Load the presentation artifact exported by the executed notebook."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return AGResults(
        schema_version=int(payload["schema_version"]),
        fitness=FitnessDefinition(
            formula=str(payload["fitness"]["formula"]),
            cv_folds=int(payload["fitness"]["cv_folds"]),
            seed=int(payload["fitness"]["seed"]),
        ),
        experiments=[
            AGExperiment(
                name=str(experiment["name"]),
                population=int(experiment["population"]),
                generations=int(experiment["generations"]),
                mutation_rate=float(experiment["mutation_rate"]),
                best_fitness=float(experiment["best_fitness"]),
                best_config=dict(experiment["best_config"]),
                history=AGHistory(
                    generation=[int(value) for value in experiment["history"]["generation"]],
                    best=[float(value) for value in experiment["history"]["best"]],
                    mean=[float(value) for value in experiment["history"]["mean"]],
                ),
            )
            for experiment in payload["experiments"]
        ],
        baseline=Baseline(
            name=str(payload["baseline"]["name"]),
            cv_fitness=float(payload["baseline"]["cv_fitness"]),
            cv_metrics={key: float(value) for key, value in payload["baseline"]["cv_metrics"].items()},
            test_metrics={key: float(value) for key, value in payload["baseline"]["test_metrics"].items()},
            false_negatives=int(payload["baseline"]["false_negatives"]),
        ),
        best_experiment=str(payload["best_experiment"]),
        best_config=dict(payload["best_config"]),
        final_comparison=dict(payload["final_comparison"]),
        subgroup_analysis=dict(payload.get("subgroup_analysis", {})),
        source=dict(payload.get("source", {})),
    )
