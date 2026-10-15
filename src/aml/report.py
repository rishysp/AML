"""Turn results.json into resume-ready bullet drafts, filled with your real numbers."""
from __future__ import annotations

from pathlib import Path

FULL = "XGBoost + network features"
NO_GRAPH = "XGBoost, no network features"
RULES = "Rules only (tuned thresholds)"


def _p(x: float) -> str:
    return f"{x * 100:.0f}%"


def write_resume_bullets(results: dict, path: Path) -> Path:
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

    rec_txt = f"{_p(best['recall_at_budget'])}"
    if rec_ci:
        rec_txt += f" (95% CI {_p(rec_ci[0])}-{_p(rec_ci[1])})"

    lines = [
        "# Resume bullet drafts (auto-generated from reports/results.json)",
        "",
        "Pick 3-4. Every number comes from your own held-out test window.",
        "",
        f"- Built an account-level AML transaction monitoring pipeline in Python on {n_tx_s} synthetic "
        f"bank transactions (IBM AML dataset), using equal-size chronological windows so thresholds and "
        f"models transfer between periods.",
        f"- At a {_p(budget)} investigator alert budget, the tuned XGBoost model caught {rec_txt} of "
        f"laundering accounts at {_p(best['precision_at_budget'])} precision, versus "
        f"{_p(rules['recall_at_budget'])} recall and {_p(rules['precision_at_budget'])} precision for "
        f"tuned rules: {ratio:.1f}x the detections for the same workload.",
        f"- Engineered network features with NetworkX and sparse matrices (Louvain communities, PageRank, "
        f"reciprocity, 3-cycles, counterparty aggregates); recall at budget "
        f"{_p(ng['recall_at_budget'])} without them vs {_p(best['recall_at_budget'])} with them"
        + (" (significant in a paired bootstrap)." if graph_sig else
           " (NOT significant in a paired bootstrap: soften or drop this claim)."),
    ]
    no_hist = m.get("XGBoost, no case-history features")
    if no_hist:
        hist_sig = diffs.get((f"{FULL} minus XGBoost, no case-history features", "recall_at_budget"),
                             {}).get("significant")
        lag = results.get("history", {}).get("label_lag_windows")
        lines.append(
            f"- Added prior-case features (exposure to accounts confirmed as laundering at least {lag} "
            f"windows earlier, mimicking investigation lag); recall at budget "
            f"{_p(no_hist['recall_at_budget'])} without them vs {_p(best['recall_at_budget'])} with them"
            + (" (significant)." if hist_sig else " (NOT significant: drop this claim).")
        )
    if seeds and stab:
        lines.append(
            f"- Validated durability with 95% bootstrap CIs, seed reruns (PR-AUC sd "
            f"{seeds['std']:.3f}), validation-vs-test stability (PR-AUC {stab['val']['pr_auc']:.2f} vs "
            f"{stab['test']['pr_auc']:.2f}) and PSI feature-drift monitoring."
        )
    lines += [
        "- Implemented structuring, rapid-movement and fan-in/fan-out scenarios with thresholds tuned "
        "against the alert budget and checked with below-the-line testing.",
        "",
        "## Sanity checks",
        "",
        "- Say 'synthetic' when describing the data.",
        "- If the validation and test PR-AUC differ a lot, mention it as a limitation, not a strength.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
