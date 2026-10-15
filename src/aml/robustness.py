"""Durability evidence: confidence intervals, paired comparisons and drift (PSI)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _recall_precision_at_budget(y: np.ndarray, s: np.ndarray, budget: float) -> tuple[float, float]:
    k = max(1, math.ceil(budget * len(y)))
    top = np.argpartition(-s, k - 1)[:k]
    tp = y[top].sum()
    return tp / max(1, y.sum()), tp / k


def bootstrap(y, scores: dict[str, np.ndarray], budget: float, n_boot: int = 200,
              seed: int = 42, pairs: list[tuple[str, str]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Account-level bootstrap. Returns (95% CIs per model, paired-difference CIs).

    The same resample is used for every model, so differences are paired:
    if a difference's interval excludes 0, the gap is unlikely to be noise.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    names = list(scores)
    rec = {n: [] for n in names}
    pre = {n: [] for n in names}
    ap = {n: [] for n in names}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        yb = y[i]
        for n in names:
            sb = np.asarray(scores[n])[i]
            r, p = _recall_precision_at_budget(yb, sb, budget)
            rec[n].append(r)
            pre[n].append(p)
            ap[n].append(average_precision_score(yb, sb))

    def ci(v):
        return np.percentile(v, 2.5), np.percentile(v, 97.5)

    rows = []
    for n in names:
        for metric, vals in [("pr_auc", ap[n]), ("recall_at_budget", rec[n]), ("precision_at_budget", pre[n])]:
            lo, hi = ci(vals)
            rows.append({"model": n, "metric": metric, "ci_low": lo, "ci_high": hi})
    diffs = []
    for a, b in pairs or []:
        for metric, d in [("pr_auc", ap), ("recall_at_budget", rec)]:
            delta = np.array(d[a]) - np.array(d[b])
            lo, hi = ci(delta)
            diffs.append({"comparison": f"{a} minus {b}", "metric": metric, "mean_diff": delta.mean(),
                          "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index. <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift."""
    e, a = expected.to_numpy(dtype=float), actual.to_numpy(dtype=float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]  # history columns are NaN where no history exists
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_table(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(train[c], test[c])} for c in columns]
    t = pd.DataFrame(rows).sort_values("psi", ascending=False)
    t["status"] = pd.cut(t["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "moderate", "major"])
    return t
