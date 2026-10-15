"""GNN experiment: directional GraphSAGE vs the XGBoost model, on the same windows and test accounts.

Run AFTER scripts/run_pipeline.py with the same config (it reuses the cached
window features, edges and the saved XGBoost test scores):

    python scripts/run_pipeline.py --config CONFIG
    python scripts/run_gnn.py      --config CONFIG          # --device cuda on a GPU

Outputs in <reports>/gnn/: results_gnn.json, model_comparison_gnn.csv,
paired_differences_gnn.csv, training_history.csv, figures/, resume_bullets_gnn.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml import evaluate, gnn, robustness, viz  # noqa: E402

XGB = "XGBoost + network features"
XGB_NG = "XGBoost, no network features"
RULES = "Rules only (tuned thresholds)"
GNN = "GraphSAGE GNN"
ENS = "Ensemble (GNN + XGBoost)"


def _rank(s: np.ndarray) -> np.ndarray:
    return pd.Series(s).rank(pct=True).to_numpy()


def write_bullets(res: dict, path: Path) -> None:
    m, diffs = res["models"], {(d["comparison"], d["metric"]): d for d in res["paired_differences"]}
    n_tx = res["data"].get("transactions")
    n_txt = f"{n_tx / 1e6:.1f}M" if n_tx else "the"
    ds = res["data"]["dataset"]
    hw = "GPU" if "cuda" in res.get("device", "") else "CPU"
    g, x, e = m[GNN], m[XGB], m[ENS]

    def sig(a):
        return diffs.get((f"{a} minus {XGB}", "recall_at_budget"), {}).get("significant") and \
            m[a]["recall_at_budget"] > x["recall_at_budget"]

    best = ENS if sig(ENS) and e["recall_at_budget"] >= g["recall_at_budget"] else (GNN if sig(GNN) else None)
    lines = ["# GNN resume bullet drafts (auto-generated from gnn/results_gnn.json)", ""]
    if best:
        b = m[best]
        lines.append(
            f"- Built a directional GraphSAGE graph neural network in PyTorch ({hw}) on {n_txt} synthetic "
            f"transactions ({ds}); {'combined with XGBoost, it' if best == ENS else 'it'} raised recall at a "
            f"1% alert budget from {x['recall_at_budget']:.0%} to {b['recall_at_budget']:.0%} "
            f"(PR-AUC {x['pr_auc']:.2f} to {b['pr_auc']:.2f}), significant in a paired bootstrap.")
    else:
        lines.append(
            f"- Benchmarked a directional GraphSAGE graph neural network in PyTorch ({hw}) against XGBoost on "
            f"{n_txt} synthetic transactions ({ds}): PR-AUC {g['pr_auc']:.2f} vs {x['pr_auc']:.2f}, with no "
            f"significant recall gain, so kept the simpler, more explainable model.")
    lines += [
        "",
        f"Recall at 1% budget: XGBoost {x['recall_at_budget']:.1%}, GNN {g['recall_at_budget']:.1%}, "
        f"ensemble {e['recall_at_budget']:.1%}.",
        f"GNN seed spread (PR-AUC sd): {res['gnn_seed_pr_auc']['std']:.4f}.",
        "",
        "Only use the 'raised recall' version if the script wrote it: it checks significance for you.",
        "A well-run benchmark that keeps the simpler model is a good interview story too.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(cfg_path: Path, device_arg: str) -> dict:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    cfg = yaml.safe_load(cfg_path.read_text())
    gcfg = cfg["gnn"]
    reports = Path(cfg["outputs"]["reports_dir"])
    reports = reports if reports.is_absolute() else ROOT / reports
    out = reports / "gnn"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    budget, target = cfg["tuning"]["alert_budget"], cfg["tuning"]["target_recall"]

    meta_path = reports / "feature_columns.json"
    if not meta_path.exists():
        sys.exit("Run scripts/run_pipeline.py with this config first (feature_columns.json not found).")
    meta = json.loads(meta_path.read_text())
    cache, names, cols = Path(meta["cache"]), meta["windows"], meta["all"]
    device = torch.device("cuda" if device_arg == "auto" and torch.cuda.is_available()
                          else ("cpu" if device_arg == "auto" else device_arg))
    log(f"device: {device}; windows: {names}; features: {len(cols)}")

    raw = {}
    for n in names:
        X = pd.read_parquet(cache / f"{n}_X.parquet").reindex(columns=cols, fill_value=0.0)
        y = pd.read_parquet(cache / f"{n}_labels.parquet")["label"]
        edges = pd.read_parquet(cache / f"{n}_edges.parquet")
        raw[n] = (X, y, edges)
    train_names = [n for n in names if n.startswith("train_")]
    mean, std = gnn.fit_scaler([gnn.transform_features(raw[n][0]) for n in train_names])
    graphs = {n: gnn.build_graph(*raw[n][:2], raw[n][2], mean, std, device) for n in names}
    del raw
    log("graphs built: " + ", ".join(f"{n} {len(graphs[n]['index']):,} nodes, "
                                     f"{graphs[n]['A_out']._nnz():,} edges" for n in names))

    # 1. Early stopping on validation to choose the number of epochs
    tr = [graphs[n] for n in train_names]
    _, best_epoch, hist = gnn.train(tr, gcfg, gcfg["seeds"][0], val_graph=graphs["val"], log=log)
    pd.DataFrame(hist).to_csv(out / "training_history.csv", index=False)
    log(f"best epoch {best_epoch}")

    # 2. Refit on train + validation for that many epochs (same protocol as XGBoost), several seeds
    fit = tr + [graphs["val"]]
    test = graphs["test"]
    yt = test["y"].cpu().numpy()
    seed_scores = []
    for s in gcfg["seeds"]:
        model, _, _ = gnn.train(fit, gcfg, s, epochs=best_epoch, log=log)
        seed_scores.append(gnn.predict(model, test))
        log(f"seed {s}: test PR-AUC {average_precision_score(yt, seed_scores[-1]):.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    gnn_score = seed_scores[0]

    # 3. Compare on the same test accounts as the XGBoost run
    xs = pd.read_parquet(reports / "test_scores.parquet").reindex(test["index"])
    if not np.array_equal(xs["label"].to_numpy(), yt):
        sys.exit("Test labels differ from the pipeline run; rerun run_pipeline.py with this config.")
    w = gcfg["ensemble_weight"]
    scores = {
        GNN: gnn_score,
        ENS: w * _rank(gnn_score) + (1 - w) * _rank(xs[XGB].to_numpy()),
        XGB: xs[XGB].to_numpy(),
        XGB_NG: xs[XGB_NG].to_numpy(),
        RULES: xs[RULES].to_numpy(),
    }
    table = evaluate.comparison_table([evaluate.summarize(n, yt, s, budget, target) for n, s in scores.items()])
    table.to_csv(out / "model_comparison_gnn.csv")
    print("\n" + table.round(4).to_string() + "\n")
    ci, diffs = robustness.bootstrap(yt, {k: scores[k] for k in [GNN, ENS, XGB]}, budget,
                                     cfg["models"]["bootstrap_reps"], cfg["models"]["random_state"],
                                     pairs=[(GNN, XGB), (ENS, XGB)])
    ci.to_csv(out / "bootstrap_ci_gnn.csv", index=False)
    diffs.to_csv(out / "paired_differences_gnn.csv", index=False)
    curves = {n: (yt, scores[n]) for n in [ENS, GNN, XGB, RULES]}
    viz.pr_curves(curves, out / "figures" / "pr_curves_gnn.png")
    viz.alert_efficiency(curves, out / "figures" / "alert_efficiency_gnn.png", budget)

    seed_pr = [float(average_precision_score(yt, s)) for s in seed_scores]
    pipe = json.loads((reports / "results.json").read_text()) if (reports / "results.json").exists() else {}
    res = {
        "data": {"dataset": Path(cfg["data"]["raw_csv"]).stem.replace("_Trans", ""),
                 "transactions": pipe.get("data", {}).get("transactions"),
                 "accounts_test": int(len(yt)), "laundering_accounts_test": int(yt.sum())},
        "gnn_config": gcfg, "best_epoch": best_epoch, "device": str(device),
        "models": json.loads(table.to_json(orient="index")),
        "paired_differences": json.loads(diffs.to_json(orient="records")),
        "confidence_intervals": json.loads(ci.to_json(orient="records")),
        "gnn_seed_pr_auc": {"values": seed_pr, "mean": float(np.mean(seed_pr)), "std": float(np.std(seed_pr))},
    }
    (out / "results_gnn.json").write_text(json.dumps(res, indent=2))
    write_bullets(res, out / "resume_bullets_gnn.md")
    print(diffs.round(4).to_string())
    log(f"done. See {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    a = ap.parse_args()
    main(Path(a.config), a.device)
