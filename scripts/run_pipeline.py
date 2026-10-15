"""End-to-end AML transaction monitoring pipeline (v2).

Usage (from the repo root):
    python scripts/run_pipeline.py                 # full run
    python scripts/run_pipeline.py --nrows 500000  # quick development run

v2 changes, all aimed at performance that holds up on new data:
- Equal-size time windows: train, validation and test account-periods are on the same scale.
- Rule thresholds tuned on the validation window, not the training data.
- New features: bursts, dwell time, incoming payment mix, neighbour aggregates.
- XGBoost random search on validation, then a refit on train + validation.
- Durability evidence: bootstrap CIs, paired significance tests, seed spread,
  validation-vs-test stability and feature drift (PSI).

Outputs land in reports/.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml import (data, evaluate, features, graph_features, history, models,  # noqa: E402
                 robustness, scenarios, tuning, viz)
from aml.report import write_resume_bullets  # noqa: E402

VERSION = 3
FULL = "XGBoost + network features"
NO_GRAPH = "XGBoost, no network features"
IFOREST = "Isolation Forest (unsupervised)"
NO_HIST = "XGBoost, no case-history features"
RULES = "Rules only (tuned thresholds)"


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_period(df: pd.DataFrame, cfg: dict) -> dict:
    """Labels, scenario scores and all features for one window."""
    labels = data.account_labels(df)
    accounts = labels.index
    thr = cfg["scenarios"]["structuring"]["reporting_threshold"]
    scen = scenarios.all_scenario_scores(df, cfg, accounts)
    g = cfg["graph"]
    base = pd.concat([
        features.account_features(df, accounts, thr),
        features.temporal_features(df, accounts),
        graph_features.graph_features(df, accounts, g["pagerank"], g["louvain"], g["louvain_seed"]),
        scen,
    ], axis=1)
    nbr = graph_features.neighbor_aggregates(df, base, g["neighbor_cols"])
    X = pd.concat([base, nbr], axis=1).astype("float32")  # float32 halves memory
    return {"labels": labels, "scenarios": scen, "X": X}


def load_or_build_period(df: pd.DataFrame, cfg: dict, cache: Path, name: str) -> dict:
    """Cache each window's features so reruns (e.g. after tuning changes) skip the slow part."""
    files = {k: cache / f"{name}_{k}.parquet" for k in ["X", "labels", "scenarios"]}
    # Edge lists are cached too: the GNN experiment (scripts/run_gnn.py) trains on them
    edges_file = cache / f"{name}_edges.parquet"
    if not edges_file.exists():
        cache.mkdir(parents=True, exist_ok=True)
        graph_features.edge_list(df).to_parquet(edges_file, index=False)
    if all(f.exists() for f in files.values()):
        return {
            "X": pd.read_parquet(files["X"]),
            "labels": pd.read_parquet(files["labels"])["label"],
            "scenarios": pd.read_parquet(files["scenarios"]),
        }
    p = build_period(df, cfg)
    cache.mkdir(parents=True, exist_ok=True)
    p["X"].to_parquet(files["X"])
    p["labels"].to_frame().to_parquet(files["labels"])
    p["scenarios"].to_parquet(files["scenarios"])
    return p


def is_network(col: str) -> bool:
    return col.startswith("g_") or col.startswith("nbr_")


def main(cfg_path: Path, nrows: int | None, reuse_search: bool = False) -> dict:
    t0 = time.time()
    cfg = yaml.safe_load(cfg_path.read_text())
    reports = Path(cfg["outputs"]["reports_dir"])
    reports = reports if reports.is_absolute() else ROOT / reports
    figs = Path(cfg["outputs"]["figures_dir"])
    figs = figs if figs.is_absolute() else ROOT / figs
    reports.mkdir(parents=True, exist_ok=True)
    budget, target = cfg["tuning"]["alert_budget"], cfg["tuning"]["target_recall"]
    mcfg = cfg["models"]
    seed = mcfg["random_state"]
    models.DEVICE = mcfg["xgboost"].get("device", "cpu")

    # Keep each earlier version's results for before/after comparisons
    prev = reports / "results.json"
    if prev.exists():
        prev_version = json.loads(prev.read_text()).get("version", 1)
        if prev_version < VERSION and not (reports / f"results_v{prev_version}.json").exists():
            shutil.copy(prev, reports / f"results_v{prev_version}.json")

    # 1. Data and equal-size windows
    d = cfg["data"]
    processed = Path(d["processed"]) if Path(d["processed"]).is_absolute() else ROOT / d["processed"]
    raw = Path(d["raw_csv"]) if Path(d["raw_csv"]).is_absolute() else ROOT / d["raw_csv"]
    if nrows:
        processed = processed.with_name(f"{processed.stem}_{nrows}{processed.suffix}")
    df = data.prepare(raw, processed, d["base_currency"], nrows or d["nrows"])
    log(f"loaded {len(df):,} transactions, laundering rate {df['is_laundering'].mean():.4%}", t0)

    n_windows = cfg["split"]["n_windows"]
    cache = reports / "cache" / f"w{n_windows}{'_n' + str(nrows) if nrows else ''}"
    data_summary = {
        "transactions": int(len(df)),
        "laundering_txn_rate": float(df["is_laundering"].mean()),
        "date_range": [str(df["ts"].min()), str(df["ts"].max())],
        "windows": n_windows,
    }
    windows = data.equal_windows(df, n_windows)
    del df
    names = [f"train_{i}" for i in range(len(windows) - 2)] + ["val", "test"]
    periods = {}
    hcfg = mcfg.get("history", {"enabled": False})
    lag = int(hcfg.get("label_lag_windows", 2))
    for i, name in enumerate(names):
        part = windows[i]
        periods[name] = load_or_build_period(part, cfg, cache, name)
        lab = periods[name]["labels"]
        if hcfg.get("enabled"):
            # Only labels from a window at least `lag` windows earlier: never this window's own
            src = names[i - lag] if i >= lag else None
            known_bad = set(periods[src]["labels"].index[periods[src]["labels"] == 1]) if src else None
            periods[name]["hist"] = history.exposure_features(part, lab.index, known_bad)
        log(f"{name}: {part['ts'].min():%m-%d %H:%M} to {part['ts'].max():%m-%d %H:%M}, "
            f"{len(part):,} txns, {len(lab):,} accounts, {int(lab.sum()):,} laundering", t0)
        windows[i] = None  # free raw transactions once the window is featurised
    del windows
    gc.collect()
    train_names = [n for n in names if n.startswith("train_")]

    # 2. Rule scenarios: thresholds tuned on validation (same window size as test)
    rules, chosen = {}, {}
    for scen_name, scen_cfg in cfg["scenarios"].items():
        grid = scen_cfg["threshold_grid"]
        sw_val = tuning.sweep(periods["val"]["scenarios"][scen_name], periods["val"]["labels"], grid)
        thr = float(tuning.pick_threshold(sw_val, budget)["threshold"])
        chosen[scen_name] = thr
        sw_test = tuning.sweep(periods["test"]["scenarios"][scen_name], periods["test"]["labels"], grid)
        btl = tuning.below_the_line(periods["test"]["scenarios"][scen_name], periods["test"]["labels"],
                                    thr, grid, cfg["tuning"]["btl_bands"])
        sw_val.to_csv(reports / f"sweep_val_{scen_name}.csv", index=False)
        sw_test.to_csv(reports / f"sweep_test_{scen_name}.csv", index=False)
        btl.to_csv(reports / f"btl_test_{scen_name}.csv", index=False)
        viz.threshold_tradeoff(sw_test, thr, scen_name, figs / f"threshold_{scen_name}.png")
        at = sw_test[sw_test["threshold"] == thr].iloc[0]
        rules[scen_name] = {
            "chosen_threshold": thr,
            "test_alerts": int(at["alerts"]),
            "test_alert_rate": float(at["alert_rate"]),
            "test_precision": float(at["precision"]),
            "test_recall": float(at["recall"]),
        }
    log(f"rule thresholds chosen on validation: {chosen}", t0)

    # 3. Matrices with identical columns everywhere
    aligned = features.align_columns({
        n: pd.concat([periods[n].pop("X")] + ([periods[n].pop("hist")] if "hist" in periods[n] else []), axis=1)
        for n in names
    })
    drop = [c for c in mcfg.get("drop_features", []) if c in aligned["test"].columns]
    aligned = {n: x.drop(columns=drop) for n, x in aligned.items()}
    cols = list(aligned["test"].columns)
    ng_cols = [c for c in cols if not is_network(c)]
    nh_cols = [c for c in cols if not c.startswith("hist_")]
    fit_names = train_names + ["val"]
    # One train+validation matrix; the training rows are a positional slice of it (no extra copy)
    X_all = pd.concat([aligned.pop(n) for n in fit_names], keys=fit_names).astype("float32")
    y_all = pd.concat([periods[n]["labels"] for n in fit_names], keys=fit_names)
    n_tr = int(sum(len(periods[n]["labels"]) for n in train_names))
    X_tr, y_tr = X_all.iloc[:n_tr], y_all.iloc[:n_tr]
    X_val, y_val = X_all.iloc[n_tr:], y_all.iloc[n_tr:]
    X_te, y_te = aligned.pop("test").astype("float32"), periods["test"]["labels"]
    yt = y_te.to_numpy()
    gc.collect()
    log(f"features: {len(cols)} total, {len(cols) - len(ng_cols)} network; "
        f"train rows {len(X_tr):,} ({int(y_tr.sum()):,} positive)", t0)

    # 4. Hyperparameter search on validation
    trials_path = reports / "xgb_search_trials.csv"
    if reuse_search and trials_path.exists():
        top = pd.read_csv(trials_path).sort_values("val_pr_auc", ascending=False).iloc[0]
        best = {k: (top[k] if k == "pos_weight_mode" else
                    (int(top[k]) if k in ("max_depth", "min_child_weight") else float(top[k])))
                for k in models.SEARCH_SPACE}
        best_iter = int(top["best_iteration"])
        log(f"reusing earlier search result (val PR-AUC {top['val_pr_auc']:.4f})", t0)
    else:
        best, best_iter, trials = models.tune_xgboost(X_tr, y_tr, X_val, y_val, mcfg["xgboost"],
                                                      mcfg["search"]["n_trials"], seed)
        trials.to_csv(trials_path, index=False)
    gc.collect()
    # The search caps rounds; re-find the best count for the chosen settings with a higher cap
    best_iter, val_score = models.find_rounds(X_tr, y_tr, X_val, y_val, best, mcfg["search"]["max_rounds"],
                                              mcfg["xgboost"]["early_stopping_rounds"], seed)
    log(f"rounds re-found with cap {mcfg['search']['max_rounds']}: best iteration {best_iter}, "
        f"val PR-AUC {val_score:.4f}", t0)
    gc.collect()
    n_final = int(max(50, best_iter) * mcfg["search"]["refit_round_factor"]) + 1
    log(f"best params {best}, best iteration {best_iter}", t0)

    # 5. Temporal stability: train-only model scored on validation and on test
    m_train_only = models.fit_final(X_tr, y_tr, best, max(50, best_iter) + 1, seed)
    stability = {}
    for split, X, y in [("val", X_val, y_val), ("test", X_te, y_te)]:
        s = m_train_only.predict_proba(X)[:, 1]
        stability[split] = evaluate.summarize(split, y.to_numpy(), s, budget, target)

    del m_train_only
    gc.collect()

    # 6. Final models refit on train + validation (each scored, then freed)
    scores = {}
    final = models.fit_final(X_all, y_all, best, n_final, seed)
    scores[FULL] = final.predict_proba(X_te)[:, 1]
    gc.collect()
    m = models.fit_final(X_all[ng_cols], y_all, best, n_final, seed)
    scores[NO_GRAPH] = m.predict_proba(X_te[ng_cols])[:, 1]
    del m
    gc.collect()
    if len(nh_cols) < len(cols):
        m = models.fit_final(X_all[nh_cols], y_all, best, n_final, seed)
        scores[NO_HIST] = m.predict_proba(X_te[nh_cols])[:, 1]
        del m
        gc.collect()
    # Isolation Forest is unsupervised and cannot use NaN history columns
    m = models.train_isolation_forest(X_tr[nh_cols], mcfg["isolation_forest"], seed)
    scores[IFOREST] = models.isolation_scores(m, X_te[nh_cols])
    del m
    gc.collect()
    scores[RULES] = models.rule_baseline_score(periods["test"]["scenarios"], chosen)
    log("final models trained", t0)
    # Saved so later experiments (GNN) are compared on exactly the same accounts and scores
    pd.DataFrame({"label": yt, **scores}, index=y_te.index).to_parquet(reports / "test_scores.parquet")
    (reports / "feature_columns.json").write_text(json.dumps(
        {"all": cols, "no_network": ng_cols, "windows": names, "cache": str(cache)}, indent=1))
    table = evaluate.comparison_table([evaluate.summarize(n, yt, s, budget, target) for n, s in scores.items()])
    table.to_csv(reports / "model_comparison_test.csv")
    print("\n" + table.round(4).to_string() + "\n")

    # 7. Durability evidence
    seed_pr = [average_precision_score(yt, scores[FULL])]
    for s in mcfg["stability_seeds"]:
        m = models.fit_final(X_all, y_all, best, n_final, s)
        seed_pr.append(average_precision_score(yt, m.predict_proba(X_te)[:, 1]))
        del m
        gc.collect()
    ci, diffs = robustness.bootstrap(yt, scores, budget, mcfg["bootstrap_reps"], seed,
                                     pairs=[(FULL, RULES), (FULL, NO_GRAPH)]
                                     + ([(FULL, NO_HIST)] if NO_HIST in scores else []))
    ci.to_csv(reports / "bootstrap_ci_test.csv", index=False)
    diffs.to_csv(reports / "paired_differences_test.csv", index=False)
    imp = pd.Series(final.feature_importances_, index=cols).sort_values(ascending=False)
    drift = robustness.psi_table(X_tr, X_te, list(imp.index[:20]))
    drift.to_csv(reports / "feature_drift_psi.csv", index=False)
    log("bootstrap, seed and drift checks done", t0)

    curves = {n: (yt, s) for n, s in scores.items()}
    viz.pr_curves(curves, figs / "pr_curves.png")
    viz.alert_efficiency(curves, figs / "alert_efficiency.png", budget)
    viz.feature_importance(final, cols, figs / "feature_importance.png")

    # 8. Results
    def ci_of(model, metric):
        r = ci[(ci["model"] == model) & (ci["metric"] == metric)].iloc[0]
        return [float(r["ci_low"]), float(r["ci_high"])]

    results = {
        "version": VERSION,
        "history": {"enabled": bool(hcfg.get("enabled")), "label_lag_windows": lag},
        "data": {**data_summary, "accounts_test": int(len(yt)), "laundering_accounts_test": int(yt.sum())},
        "config": {"alert_budget": budget, "target_recall": target},
        "rules": rules,
        "xgb_best_params": {k: (v if isinstance(v, str) else float(v)) for k, v in best.items()},
        "xgb_rounds": n_final,
        "models": json.loads(table.to_json(orient="index")),
        "confidence_intervals": {
            m: {k: ci_of(m, k) for k in ["pr_auc", "recall_at_budget", "precision_at_budget"]}
            for m in scores
        },
        "paired_differences": json.loads(diffs.to_json(orient="records")),
        "seed_stability_pr_auc": {"values": [float(v) for v in seed_pr],
                                  "mean": float(np.mean(seed_pr)), "std": float(np.std(seed_pr))},
        "temporal_stability_train_only_model": stability,
        "drift_top_features": json.loads(drift.head(10).to_json(orient="records")),
        "top_features": [str(c) for c in imp.index[:10]],
    }
    (reports / "results.json").write_text(json.dumps(results, indent=2))

    # Version history: one row per metric, one column per version
    runs = {f"v{v}": json.loads((reports / f"results_v{v}.json").read_text())
            for v in range(1, VERSION) if (reports / f"results_v{v}.json").exists()}
    runs[f"v{VERSION}"] = results
    rows = []
    for mname in [FULL, RULES]:
        for k in ["pr_auc", "recall_at_budget", "precision_at_budget"]:
            row = {"model": mname, "metric": k}
            row.update({v: r["models"].get(mname, {}).get(k) for v, r in runs.items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(reports / "version_history.csv", index=False)

    write_resume_bullets(results, reports / "resume_bullets.md")
    log(f"done. See {reports}", t0)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--nrows", type=int, default=None, help="limit rows for a quick run")
    ap.add_argument("--reuse-search", action="store_true",
                    help="skip the hyperparameter search and reuse reports/xgb_search_trials.csv")
    a = ap.parse_args()
    main(Path(a.config), a.nrows, a.reuse_search)
