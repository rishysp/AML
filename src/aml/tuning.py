"""Threshold tuning and below-the-line (BTL) testing for rule scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sweep(score: pd.Series, labels: pd.Series, grid: list[float]) -> pd.DataFrame:
    """Alert volume, precision and recall for each candidate threshold."""
    score = score.reindex(labels.index).fillna(0.0)
    n, n_pos = len(labels), int(labels.sum())
    rows = []
    for thr in grid:
        alerted = score >= thr
        alerts = int(alerted.sum())
        tp = int((alerted & (labels == 1)).sum())
        rows.append({
            "threshold": thr,
            "alerts": alerts,
            "alert_rate": alerts / n if n else 0.0,
            "true_positives": tp,
            "precision": tp / alerts if alerts else 0.0,
            "recall": tp / n_pos if n_pos else 0.0,
        })
    return pd.DataFrame(rows)


def pick_threshold(sweep_df: pd.DataFrame, alert_budget: float) -> pd.Series:
    """Highest-recall threshold whose alert rate fits investigator capacity.

    Falls back to the strictest threshold if nothing fits the budget.
    """
    fits = sweep_df[sweep_df["alert_rate"] <= alert_budget]
    if fits.empty:
        return sweep_df.sort_values("alert_rate").iloc[0]
    return fits.sort_values(["recall", "precision"], ascending=False).iloc[0]


def below_the_line(score: pd.Series, labels: pd.Series, threshold: float,
                   grid: list[float], n_bands: int = 3) -> pd.DataFrame:
    """Laundering rate in the bands just below the chosen threshold.

    BTL testing asks: are we missing productive activity just under the line?
    A high rate in the first band below is evidence the threshold is too high.
    """
    score = score.reindex(labels.index).fillna(0.0)
    lower = sorted(t for t in grid if t < threshold)[-n_bands:]
    edges = lower + [threshold]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_band = (score >= lo) & (score < hi)
        k = int(in_band.sum())
        rows.append({
            "band": f"[{lo}, {hi})",
            "accounts": k,
            "laundering_accounts": int(labels[in_band].sum()),
            "laundering_rate": float(labels[in_band].mean()) if k else np.nan,
        })
    above = score >= threshold
    rows.append({
        "band": f">= {threshold} (alerted)",
        "accounts": int(above.sum()),
        "laundering_accounts": int(labels[above].sum()),
        "laundering_rate": float(labels[above].mean()) if above.any() else np.nan,
    })
    return pd.DataFrame(rows)
