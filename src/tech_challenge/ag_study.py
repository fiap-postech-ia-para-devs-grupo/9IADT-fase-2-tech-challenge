from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from tech_challenge.paths import AG_EXPERIMENT_RESULTS, BEST_MODEL

SEED = 42
FITNESS_FORMULA = "0.6 * f1 + 0.4 * recall"
CV_FOLDS = 3

RF_BASELINE_CONFIG: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": None,
}

GENE_SPACE: dict[str, list[Any]] = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [4, 8, 12, 16, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": [None, "balanced"],
}

EXPERIMENTS = [
    {"name": "Exp1_Referencia", "population": 15, "generations": 15, "mutation_rate": 0.10},
    {"name": "Exp2_Populacao_Maior", "population": 30, "generations": 15, "mutation_rate": 0.10},
    {"name": "Exp3_Mutacao_Maior", "population": 30, "generations": 15, "mutation_rate": 0.30},
]


@dataclass
class StudyData:
    x: pd.DataFrame
    y: pd.Series
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    x_train_scaled: pd.DataFrame
    x_test_scaled: pd.DataFrame
    scaler: StandardScaler


def prepare_data() -> StudyData:
    dataset = load_breast_cancer()
    x = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = 1 - pd.Series(dataset.target, name="target")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=SEED)
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x.columns, index=x_test.index)
    return StudyData(
        x=x,
        y=y,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        x_train_scaled=x_train_scaled,
        x_test_scaled=x_test_scaled,
        scaler=scaler,
    )


def build_model(genes: dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(**genes, random_state=SEED, n_jobs=-1)


class GeneticOptimizer:
    def __init__(self, data: StudyData) -> None:
        self.data = data
        splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        self.cv_splits = list(splitter.split(data.x_train_scaled, data.y_train))
        self.cache: dict[tuple[tuple[str, Any], ...], tuple[float, dict[str, float]]] = {}

    def fitness(self, genes: dict[str, Any]) -> tuple[float, dict[str, float]]:
        key = tuple(sorted(genes.items()))
        if key in self.cache:
            return self.cache[key]

        f1_values: list[float] = []
        recall_values: list[float] = []
        for train_index, validation_index in self.cv_splits:
            model = build_model(genes)
            model.fit(
                self.data.x_train_scaled.iloc[train_index],
                self.data.y_train.iloc[train_index],
            )
            prediction = model.predict(self.data.x_train_scaled.iloc[validation_index])
            expected = self.data.y_train.iloc[validation_index]
            f1_values.append(f1_score(expected, prediction, zero_division=0))
            recall_values.append(recall_score(expected, prediction, zero_division=0))

        metrics = {
            "f1": float(np.mean(f1_values)),
            "recall": float(np.mean(recall_values)),
        }
        score = 0.6 * metrics["f1"] + 0.4 * metrics["recall"]
        self.cache[key] = (float(score), metrics)
        return self.cache[key]

    @staticmethod
    def random_individual(rng: random.Random) -> dict[str, Any]:
        return {name: rng.choice(values) for name, values in GENE_SPACE.items()}

    @staticmethod
    def crossover(parent_a: dict[str, Any], parent_b: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        return {name: rng.choice([parent_a[name], parent_b[name]]) for name in GENE_SPACE}

    @staticmethod
    def mutate(individual: dict[str, Any], mutation_rate: float, rng: random.Random) -> dict[str, Any]:
        child = dict(individual)
        if rng.random() < mutation_rate:
            gene = rng.choice(list(GENE_SPACE))
            child[gene] = rng.choice(GENE_SPACE[gene])
        return child

    @staticmethod
    def tournament(
        scored_population: list[tuple[dict[str, Any], float]], rng: random.Random, size: int = 3
    ) -> dict[str, Any]:
        return max(rng.sample(scored_population, size), key=lambda item: item[1])[0]

    def run(self, config: dict[str, Any], seed: int, verbose: bool = True) -> dict[str, Any]:
        rng = random.Random(seed)
        population = [self.random_individual(rng) for _ in range(config["population"])]
        history: list[dict[str, float | int]] = []
        best_overall: tuple[dict[str, Any] | None, float, dict[str, float]] = (None, -np.inf, {})
        started_at = time.perf_counter()

        for generation in range(config["generations"]):
            scored = [(individual, *self.fitness(individual)) for individual in population]
            scored.sort(key=lambda item: item[1], reverse=True)
            best_individual, best_score, best_metrics = scored[0]
            mean_score = float(np.mean([score for _, score, _ in scored]))
            history.append({"generation": generation, "best": best_score, "mean": mean_score})
            if best_score > best_overall[1]:
                best_overall = (dict(best_individual), best_score, best_metrics)
            if verbose:
                print(f"Geração {generation:02d} | melhor={best_score:.4f} | média={mean_score:.4f}")

            next_population = [dict(item[0]) for item in scored[:2]]
            selectable = [(item[0], item[1]) for item in scored]
            while len(next_population) < config["population"]:
                parent_a = self.tournament(selectable, rng)
                parent_b = self.tournament(selectable, rng)
                child = self.crossover(parent_a, parent_b, rng)
                next_population.append(self.mutate(child, config["mutation_rate"], rng))
            population = next_population

        best_config, best_fitness, best_metrics = best_overall
        return {
            "name": config["name"],
            "population": config["population"],
            "generations": config["generations"],
            "mutation_rate": config["mutation_rate"],
            "best_fitness": float(best_fitness),
            "best_config": best_config,
            "best_cv_metrics": best_metrics,
            "duration_seconds": time.perf_counter() - started_at,
            "history": {
                "generation": [int(item["generation"]) for item in history],
                "best": [float(item["best"]) for item in history],
                "mean": [float(item["mean"]) for item in history],
            },
        }


def evaluate_on_test(data: StudyData, genes: dict[str, Any]) -> dict[str, Any]:
    model = build_model(genes)
    model.fit(data.x_train_scaled, data.y_train)
    prediction = model.predict(data.x_test_scaled)
    probability = model.predict_proba(data.x_test_scaled)[:, 1]
    matrix = confusion_matrix(data.y_test, prediction)
    return {
        "model": model,
        "prediction": prediction,
        "probability": probability,
        "confusion_matrix": matrix,
        "false_negatives": int(matrix[1, 0]),
        "metrics": {
            "accuracy": float(accuracy_score(data.y_test, prediction)),
            "precision": float(precision_score(data.y_test, prediction)),
            "recall": float(recall_score(data.y_test, prediction)),
            "f1": float(f1_score(data.y_test, prediction)),
            "roc_auc": float(roc_auc_score(data.y_test, probability)),
        },
    }


def subgroup_analysis(data: StudyData, prediction: np.ndarray) -> dict[str, Any]:
    median = float(data.x_train["mean radius"].median())
    group = np.where(data.x_test["mean radius"] <= median, "menor_ou_igual_mediana", "maior_que_mediana")
    recalls: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in np.unique(group):
        mask = group == name
        recalls[str(name)] = float(recall_score(data.y_test[mask], prediction[mask], zero_division=0))
        counts[str(name)] = int(mask.sum())
    return {"feature": "mean radius", "training_median": median, "recall": recalls, "sample_count": counts}


def run_study(
    *,
    model_path: Path = BEST_MODEL,
    results_path: Path = AG_EXPERIMENT_RESULTS,
    verbose: bool = True,
) -> dict[str, Any]:
    data = prepare_data()
    optimizer = GeneticOptimizer(data)
    baseline_cv_fitness, baseline_cv_metrics = optimizer.fitness(RF_BASELINE_CONFIG)
    experiments = []
    for index, config in enumerate(EXPERIMENTS):
        if verbose:
            print(f"\n--- {config['name']} ---")
        experiments.append(optimizer.run(config, seed=SEED + index, verbose=verbose))

    winner = max(experiments, key=lambda experiment: experiment["best_fitness"])
    baseline_test = evaluate_on_test(data, RF_BASELINE_CONFIG)
    optimized_test = evaluate_on_test(data, winner["best_config"])
    delta = {
        "f1": optimized_test["metrics"]["f1"] - baseline_test["metrics"]["f1"],
        "recall": optimized_test["metrics"]["recall"] - baseline_test["metrics"]["recall"],
        "false_negative_reduction": baseline_test["false_negatives"] - optimized_test["false_negatives"],
    }
    artifact = {
        "schema_version": 2,
        "source": {
            "notebook": "notebooks/tech_challenge_fase2.ipynb",
            "description": "Artefato exportado automaticamente pelo estudo executado.",
        },
        "fitness": {"formula": FITNESS_FORMULA, "cv_folds": CV_FOLDS, "seed": SEED},
        "baseline": {
            "name": "RF_Baseline",
            "cv_fitness": baseline_cv_fitness,
            "cv_metrics": baseline_cv_metrics,
            "test_metrics": baseline_test["metrics"],
            "false_negatives": baseline_test["false_negatives"],
        },
        "experiments": experiments,
        "best_experiment": winner["name"],
        "best_config": winner["best_config"],
        "final_comparison": {
            "baseline": {"metrics": baseline_test["metrics"], "false_negatives": baseline_test["false_negatives"]},
            "optimized": {
                "metrics": optimized_test["metrics"],
                "false_negatives": optimized_test["false_negatives"],
            },
            "delta": delta,
        },
        "subgroup_analysis": subgroup_analysis(data, optimized_test["prediction"]),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": optimized_test["model"],
            "scaler": data.scaler,
            "features": list(data.x.columns),
            "genes": winner["best_config"],
        },
        model_path,
    )
    results_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "data": data,
        "artifact": artifact,
        "baseline_test": baseline_test,
        "optimized_test": optimized_test,
        "model_path": model_path,
        "results_path": results_path,
    }
