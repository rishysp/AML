"""Smoke tests on a small synthetic dataset in the IBM AML CSV format.

Run with:  pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aml import data, evaluate, scenarios, tuning  # noqa: E402

RAW_COLUMNS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received",
               "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]


def make_synthetic(n_normal: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2022-09-01")
    accts = [f"A{i:04d}" for i in range(400)]
    rows = []
    for _ in range(n_normal):
        a, b = rng.choice(accts, 2, replace=False)
        amt = float(np.round(rng.lognormal(6, 1.2), 2))
        ts = start + pd.Timedelta(minutes=int(rng.integers(0, 20 * 24 * 60)))
        cur, fx = ("Euro", 0.9) if rng.random() < 0.1 else ("US Dollar", 1.0)
        rows.append([ts, "10", a, "20", b, amt * fx, cur, amt, "US Dollar", "ACH", 0])
        if cur == "Euro":  # some reverse-direction rows so the EUR rate is derivable both ways
            rows.append([ts, "20", b, "10", a, amt, "US Dollar", amt * fx, "Euro", "Wire", 0])
    # Planted structuring: many payments just under 10,000
    for k in range(30):
        ts = start + pd.Timedelta(days=int(k % 20), hours=k % 24)
        rows.append([ts, "10", "S0001", "20", f"M{k % 5}", 9500.0, "US Dollar", 9500.0,
                     "US Dollar", "Cash", 1])
    # Planted fan-in collection account plus a 3-cycle
    for k in range(40):
        ts = start + pd.Timedelta(days=int(k % 20), hours=3)
        rows.append([ts, "30", f"F{k}", "40", "C0001", 900.0, "US Dollar", 900.0, "US Dollar", "Wire", 1])
    for d in range(0, 20, 2):
        ts = start + pd.Timedelta(days=d)
        for src, dst in [("L1", "L2"), ("L2", "L3"), ("L3", "L1")]:
            rows.append([ts, "50", src, "50", dst, 20000.0, "US Dollar", 20000.0, "US Dollar", "Wire", 1])
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y/%m/%d %H:%M")
    return df


def test_data_and_scenarios(tmp_path):
    raw = tmp_path / "raw.csv"
    make_synthetic().to_csv(raw, index=False)
    df = data.prepare(raw, tmp_path / "proc.parquet")
    assert {"src", "dst", "amount_usd"}.issubset(df.columns)
    assert df["amount_usd"].notna().all()

    labels = data.account_labels(df)
    assert labels.loc["10_S0001"] == 1 and labels.loc["40_C0001"] == 1

    s = scenarios.structuring_score(df, 10000, 0.9)
    assert s.loc["10_S0001"] == 30
    f = scenarios.fan_score(df)
    assert f.loc["40_C0001"] == 40

    sw = tuning.sweep(s, labels, [1, 5, 10])
    assert (sw["alerts"].diff().dropna() <= 0).all()  # stricter threshold, fewer alerts


def test_history_features():
    from aml import history

    df = pd.DataFrame({"src": ["a", "a", "b", "c"], "dst": ["b", "x", "c", "c"],
                       "amount_usd": [100.0, 50.0, 20.0, 5.0]})
    accts = pd.Index(["a", "b", "c", "x"])
    none = history.exposure_features(df, accts, None)
    assert none.isna().all().all()  # no history yet -> unknown, not "clean"
    h = history.exposure_features(df, accts, {"x"})
    assert h.loc["x", "hist_flagged_before"] == 1 and h.loc["a", "hist_flagged_before"] == 0
    assert h.loc["a", "hist_n_bad_cp"] == 1 and h.loc["a", "hist_share_bad_cp"] == 0.5
    assert h.loc["a", "hist_usd_with_bad"] == 50.0
    assert h.loc["c", "hist_n_bad_cp"] == 0  # self-transfer ignored


def test_metrics():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    score = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.1, 0.05, 0.3, 0.0, 0.6])
    assert evaluate.alerts_for_recall(y, score, 1.0) == 4
    assert evaluate.at_budget(y, score, 0.2)["recall"] == 1 / 3


def test_end_to_end(tmp_path, monkeypatch):
    """Full pipeline on synthetic data, with outputs redirected to a temp folder."""
    import run_pipeline

    raw = tmp_path / "raw.csv"
    make_synthetic(n_normal=6000).to_csv(raw, index=False)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["data"]["raw_csv"] = str(raw)
    cfg["data"]["processed"] = str(tmp_path / "proc.parquet")
    cfg["outputs"] = {"reports_dir": str(tmp_path / "reports"), "figures_dir": str(tmp_path / "reports/fig")}
    cfg["models"]["xgboost"]["n_estimators"] = 50
    cfg["models"]["isolation_forest"]["n_estimators"] = 50
    cfg["models"]["search"]["n_trials"] = 2
    cfg["models"]["stability_seeds"] = [7]
    cfg["models"]["bootstrap_reps"] = 20
    cfg["split"]["n_windows"] = 4
    cfg["tuning"]["alert_budget"] = 0.05
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    results = run_pipeline.main(cfg_path, None)
    assert "XGBoost + network features" in results["models"]
    assert (tmp_path / "reports" / "resume_bullets.md").exists()
    assert (tmp_path / "reports" / "fig" / "pr_curves.png").exists()

    # GNN experiment on the same cached windows (skipped if PyTorch is not installed)
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    import run_gnn

    cfg["gnn"].update(max_epochs=6, patience=3, seeds=[42, 7], hidden=16)
    cfg_path.write_text(yaml.safe_dump(cfg))
    res = run_gnn.main(cfg_path, "cpu")
    assert "GraphSAGE GNN" in res["models"] and "Ensemble (GNN + XGBoost)" in res["models"]
    assert (tmp_path / "reports" / "gnn" / "resume_bullets_gnn.md").exists()
