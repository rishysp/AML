"""Metrics framed the way a monitoring team reads them: alerts, recall and workload."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def at_budget(y: np.ndarray, score: np.ndarray, budget: float) -> dict:
    """Precision and recall if investigators review the top `budget` share of accounts."""
    y, score = np.asarray(y), np.asarray(score)
    k = max(1, math.ceil(budget * len(y)))
    top = np.argsort(-score, kind="stable")[:k]
    tp = int(y[top].sum())
    return {"alerts": k, "precision": tp / k, "recall": tp / max(1, int(y.sum()))}


def alerts_for_recall(y: np.ndarray, score: np.ndarray, target: float) -> int:
    """Smallest number of alerts that catches `target` share of positive accounts."""
    y, score = np.asarray(y), np.asarray(score)
    order = np.argsort(-score, kind="stable")
    caught = np.cumsum(y[order])
    needed = math.ceil(target * y.sum())
    hit = np.nonzero(caught >= needed)[0]
    return int(hit[0] + 1) if len(hit) else len(y)


def summarize(name: str, y, score, budget: float, target_recall: float) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    b = at_budget(y, score, budget)
    return {
        "model": name,
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan"),
        "precision_at_budget": b["precision"],
        "recall_at_budget": b["recall"],
        "alerts_at_budget": b["alerts"],
        f"alerts_for_{int(target_recall * 100)}pct_recall": alerts_for_recall(y, score, target_recall),
    }


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("model").sort_values("pr_auc", ascending=False)
