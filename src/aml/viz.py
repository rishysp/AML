"""Charts for presenting monitoring results to a non-technical manager."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

COLORS = ["#1f5fa8", "#d9822b", "#2e8b57", "#8e44ad", "#7f8c8d"]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def threshold_tradeoff(sweep_df: pd.DataFrame, chosen: float, scenario: str, path: Path) -> Path:
    """Alert volume (bars) against recall and precision (lines) for each threshold."""
    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = np.arange(len(sweep_df))
    ax1.bar(x, sweep_df["alerts"], color="#c9d6e8", label="Alerts")
    ax1.set_ylabel("Alerts generated")
    ax1.set_xticks(x, [f"{t:g}" for t in sweep_df["threshold"]])
    ax1.set_xlabel("Threshold")
    ax2 = ax1.twinx()
    ax2.plot(x, sweep_df["recall"], "o-", color=COLORS[0], label="Recall")
    ax2.plot(x, sweep_df["precision"], "s--", color=COLORS[1], label="Precision")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Rate")
    if chosen in set(sweep_df["threshold"]):
        pos = int(np.flatnonzero(sweep_df["threshold"].to_numpy() == chosen)[0])
        ax1.axvline(pos, color="black", lw=1, ls=":")
        ax1.text(pos, ax1.get_ylim()[1] * 0.95, " chosen", va="top")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper right", frameon=False)
    ax1.set_title(f"{scenario.replace('_', ' ').title()}: threshold trade-off")
    return _save(fig, path)


def pr_curves(results: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for (name, (y, s)), color in zip(results.items(), COLORS):
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=color, label=name)
    base = np.mean(next(iter(results.values()))[0])
    ax.axhline(base, color="grey", lw=1, ls=":", label=f"Base rate ({base:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on held-out test period")
    ax.legend(frameon=False)
    return _save(fig, path)


def alert_efficiency(results: dict[str, tuple[np.ndarray, np.ndarray]], path: Path,
                     budget: float | None = None) -> Path:
    """Share of laundering accounts caught vs share of accounts alerted: the workload view."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for (name, (y, s)), color in zip(results.items(), COLORS):
        y = np.asarray(y)
        order = np.argsort(-np.asarray(s), kind="stable")
        caught = np.cumsum(y[order]) / max(1, y.sum())
        alerted = np.arange(1, len(y) + 1) / len(y)
        ax.plot(alerted, caught, color=color, label=name)
    if budget:
        ax.axvline(budget, color="black", lw=1, ls=":", label=f"Alert budget ({budget:.1%})")
    ax.set_xscale("log")
    ax.set_xlabel("Share of accounts alerted (log scale)")
    ax.set_ylabel("Share of laundering accounts caught")
    ax.set_title("Detection vs investigator workload")
    ax.legend(frameon=False)
    return _save(fig, path)


def feature_importance(model, feature_names: list[str], path: Path, top: int = 15) -> Path:
    imp = pd.Series(model.feature_importances_, index=feature_names).nlargest(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [COLORS[2] if n.startswith("g_") else COLORS[0] for n in imp.index]
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_title("Top features (green = network features)")
    ax.set_xlabel("Importance (gain share)")
    return _save(fig, path)
