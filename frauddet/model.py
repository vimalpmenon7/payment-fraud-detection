"""Training, model selection, cost-based thresholding and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, build_features

REVIEW_COST = 5.0  # cost of manually reviewing one flagged transaction


def time_split(df: pd.DataFrame, train: float = 0.7, valid: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split (train < valid < test) so the test set simulates the future."""
    if not (0 < train < 1 and 0 < valid < 1 and train + valid < 1):
        raise ValueError("invalid split fractions")
    n = len(df)
    a, b = int(n * train), int(n * (train + valid))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def expected_cost(y: np.ndarray, amount: np.ndarray, flagged: np.ndarray, review_cost: float = REVIEW_COST) -> float:
    """Missed-fraud loss (the transaction amount) plus a fixed review cost for every flagged transaction."""
    missed = ((y == 1) & ~flagged) * amount
    return float(missed.sum() + review_cost * flagged.sum())


def best_threshold(y: np.ndarray, amount: np.ndarray, proba: np.ndarray, review_cost: float = REVIEW_COST) -> float:
    """Threshold on predicted probability that minimises expected cost (searched over observed scores)."""
    candidates = np.unique(np.concatenate([np.quantile(proba, np.linspace(0.5, 0.999, 120)), [0.5]]))
    costs = [expected_cost(y, amount, proba >= t, review_cost) for t in candidates]
    return float(candidates[int(np.argmin(costs))])


def candidate_models(seed: int = 0) -> dict[str, Any]:
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, random_state=seed),
    }


def metrics_at(y: np.ndarray, amount: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    flagged = proba >= threshold
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "precision": round(float(precision_score(y, flagged, zero_division=0)), 4),
        "recall": round(float(recall_score(y, flagged, zero_division=0)), 4),
        "flagged_rate": round(float(flagged.mean()), 4),
        "expected_cost": round(expected_cost(y, amount, flagged), 2),
        "cost_if_no_model": round(expected_cost(y, amount, np.zeros_like(flagged)), 2),
    }


@dataclass
class TrainResult:
    model_name: str
    threshold: float
    validation: dict[str, dict[str, float]]
    test: dict[str, float]
    base_rate_test: float
    top_features: list[tuple[str, float]]


def train(df: pd.DataFrame, seed: int = 0) -> tuple[Any, TrainResult]:
    """Fit candidates on the train period, pick by validation PR-AUC, tune threshold on validation, report on test."""
    feats = build_features(df)
    tr, va, te = time_split(df)
    X_tr, X_va, X_te = feats.loc[tr.index], feats.loc[va.index], feats.loc[te.index]
    y_tr, y_va, y_te = tr["is_fraud"].to_numpy(), va["is_fraud"].to_numpy(), te["is_fraud"].to_numpy()
    if y_tr.sum() < 5 or y_va.sum() < 5 or y_te.sum() < 5:
        raise ValueError("Too few fraud examples in a split; generate more data")

    validation: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}
    for name, model in candidate_models(seed).items():
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, 1]
        validation[name] = {"roc_auc": round(float(roc_auc_score(y_va, p)), 4),
                            "pr_auc": round(float(average_precision_score(y_va, p)), 4)}
        fitted[name] = model

    best = max(validation, key=lambda k: validation[k]["pr_auc"])
    model = fitted[best]
    threshold = best_threshold(y_va, va["amount"].to_numpy(), model.predict_proba(X_va)[:, 1])
    p_te = model.predict_proba(X_te)[:, 1]
    test = metrics_at(y_te, te["amount"].to_numpy(), p_te, threshold)

    imp = permutation_importance(model, X_te, y_te, scoring="average_precision", n_repeats=3, random_state=seed)
    order = np.argsort(imp.importances_mean)[::-1][:5]
    top = [(FEATURES[i], round(float(imp.importances_mean[i]), 4)) for i in order]
    return model, TrainResult(best, round(threshold, 4), validation, test, round(float(y_te.mean()), 4), top)


def save(model: Any, result: TrainResult, directory: str | Path) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": result.threshold, "features": FEATURES}, d / "model.joblib")
    (d / "report.json").write_text(json.dumps(result.__dict__, indent=2, default=list), encoding="utf-8")


def load(directory: str | Path) -> dict[str, Any]:
    bundle = joblib.load(Path(directory) / "model.joblib")
    if bundle.get("features") != FEATURES:
        raise ValueError("Model was trained with different features")
    return bundle


def score(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Score transactions (sorted by ts, with history in ``df``). Returns txn_id, fraud_probability, flagged."""
    proba = bundle["model"].predict_proba(build_features(df))[:, 1]
    return pd.DataFrame({"txn_id": df["txn_id"].to_numpy(), "fraud_probability": proba.round(4),
                         "flagged": proba >= bundle["threshold"]})
