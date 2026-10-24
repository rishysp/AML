"""Plain-language summary of a run's key findings, generated from results.json."""
from __future__ import annotations

from pathlib import Path

FULL = "XGBoost + network features"
NO_GRAPH = "XGBoost, no network features"
NO_HIST = "XGBoost, no case-history features"
RULES = "Rules only (tuned thresholds)"


def _p(x: float) -> str:
    return f"{x * 100:.0f}%"


def write_summary(results: dict, path: Path) -> Path:
    """Write `summary.md`. Claims are worded by their statistical support, never stronger."""
    m = results["models"]
    best, ng, rules = m[FULL], m[NO_GRAPH], m[RULES]
    budget = results["config"]["alert_budget"]
    n_tx = results["data"]["transactions"]
    n_tx_s = f"{n_tx / 1e6:.1f}M" if n_tx >= 1e6 else f"{n_tx:,}"
    ci = results.get("confidence_intervals", {}).get(FULL, {})
    rec_ci = ci.get("recall_at_budget")
    ratio = best["recall_at_budget"] / max(1e-9, rules["recall_at_budget"])
    diffs = {(d["comparison"], d["metric"]): d for d in results.get("paired_differences", [])}
    graph_sig = diffs.get((f"{FULL} minus {NO_GRAPH}", "recall_at_budget"), {}).get("significant")
    seeds = results.get("seed_stability_pr_auc", {})
    stab = results.get("temporal_stability_train_only_model", {})

    rec_txt = _p(best["recall_at_budget"])
    if rec_ci:
        rec_txt += f" (95% CI {_p(rec_ci[0])}-{_p(rec_ci[1])})"

    lines = [
        "# Key findings (generated from results.json)",
        "",
        f"- **Data:** {n_tx_s} synthetic bank transactions (IBM AML dataset), split into equal-size "
        f"chronological windows; results are for the held-out test window.",
        f"- **Model vs rules:** at a {_p(budget)} investigator alert budget, XGBoost caught {rec_txt} of "
        f"laundering accounts at {_p(best['precision_at_budget'])} precision, versus "
        f"{_p(rules['recall_at_budget'])} recall and {_p(rules['precision_at_budget'])} precision for "
        f"tuned rules: {ratio:.1f}x the detections for the same workload.",
        f"- **Network features:** recall at budget {_p(ng['recall_at_budget'])} without them vs "
        f"{_p(best['recall_at_budget'])} with them"
        + (" (significant in a paired bootstrap)." if graph_sig else
           " (not significant in a paired bootstrap)."),
    ]
    no_hist = m.get(NO_HIST)
    if no_hist:
        hist_sig = diffs.get((f"{FULL} minus {NO_HIST}", "recall_at_budget"), {}).get("significant")
        lag = results.get("history", {}).get("label_lag_windows")
        lines.append(
            f"- **Prior-case features** (exposure to accounts confirmed at least {lag} windows earlier): "
            f"recall at budget {_p(no_hist['recall_at_budget'])} without them vs "
            f"{_p(best['recall_at_budget'])} with them"
            + (" (significant)." if hist_sig else " (not significant).")
        )
    if seeds and stab:
        lines.append(
            f"- **Stability:** PR-AUC SD across seeds {seeds['std']:.4f}; a model trained only on the "
            f"training windows scored PR-AUC {stab['val']['pr_auc']:.2f} on validation and "
            f"{stab['test']['pr_auc']:.2f} on test."
        )
    lines += [
        "",
        "Details: `model_comparison_test.csv`, `paired_differences_test.csv`, `bootstrap_ci_test.csv`, "
        "`figures/`. A large validation-test gap is a limitation to investigate, not a strength.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
