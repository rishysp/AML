"""Account-level behavioural features, computed only from one period's transactions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _side_stats(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    g = df.groupby(key)["amount_usd"]
    stats = pd.DataFrame({
        f"{prefix}_count": g.size(),
        f"{prefix}_sum": g.sum(),
        f"{prefix}_mean": g.mean(),
        f"{prefix}_max": g.max(),
        f"{prefix}_std": g.std().fillna(0.0),
    })
    stats.index.name = "account"
    return stats


def account_features(df: pd.DataFrame, accounts: pd.Index, reporting_threshold: float) -> pd.DataFrame:
    """One row per account. Every feature is explainable to an investigator."""
    out_stats = _side_stats(df, "src", "out")
    in_stats = _side_stats(df, "dst", "in")

    # Vectorised flags, then one groupby (fast on millions of rows)
    hour = df["ts"].dt.hour
    flags = df.assign(
        cross_currency=(df["curr_paid"] != df["curr_recv"]).astype(float),
        cross_bank=(df["src_bank"] != df["dst_bank"]).astype(float),
        # Roundness is checked in the original currency, before FX conversion
        round_amount=(np.round(df["amount_paid"]) % 1000 == 0).astype(float),
        near_threshold=((df["amount_usd"] >= 0.9 * reporting_threshold)
                        & (df["amount_usd"] < reporting_threshold)).astype(float),
        night=((hour < 6) | (hour >= 22)).astype(float),
        day=df["ts"].dt.normalize(),
    )
    g = flags.groupby("src")
    extra = pd.DataFrame({
        "n_unique_dst": g["dst"].nunique(),
        "cross_currency_share": g["cross_currency"].mean(),
        "cross_bank_share": g["cross_bank"].mean(),
        "round_amount_share": g["round_amount"].mean(),
        "near_threshold_count": g["near_threshold"].sum(),
        "night_share": g["night"].mean(),
        "active_days_out": g["day"].nunique(),
    })
    extra.index.name = "account"
    in_unique = df.groupby("dst")["src"].nunique().rename("n_unique_src")
    in_unique.index.name = "account"

    # Payment-format mix of outgoing transactions (wire, ACH, cash, cheque, ...)
    fmt = pd.crosstab(df["src"], df["payment_format"], normalize="index")
    fmt.columns = [f"fmt_{str(c).lower().replace(' ', '_')}" for c in fmt.columns]
    fmt.index.name = "account"

    feats = pd.concat([out_stats, in_stats, extra, in_unique, fmt], axis=1)
    feats = feats.reindex(accounts).fillna(0.0)
    feats["passthrough_ratio"] = feats["out_sum"] / (feats["in_sum"] + 1.0)
    feats["self_transfer"] = 0.0
    self_tx = df.loc[df["src"] == df["dst"], "src"].value_counts()
    feats.loc[feats.index.intersection(self_tx.index), "self_transfer"] = self_tx
    return feats


def temporal_features(df: pd.DataFrame, accounts: pd.Index) -> pd.DataFrame:
    """Bursts, activity span and dwell time (how fast money leaves after it arrives)."""
    hour = df["ts"].dt.floor("h")
    day = df["ts"].dt.floor("D")
    out = pd.DataFrame({
        "burst_out_1h": df.groupby(["src", hour]).size().groupby(level=0).max(),
        "burst_out_1d": df.groupby(["src", day]).size().groupby(level=0).max(),
    })
    inn = pd.DataFrame({
        "burst_in_1h": df.groupby(["dst", hour]).size().groupby(level=0).max(),
        "burst_in_1d": df.groupby(["dst", day]).size().groupby(level=0).max(),
        "n_unique_curr_in": df.groupby("dst")["curr_recv"].nunique(),
    })
    out["n_unique_curr_out"] = df.groupby("src")["curr_paid"].nunique()
    out.index.name = inn.index.name = "account"

    both = pd.concat([df[["src", "ts"]].rename(columns={"src": "account"}),
                      df[["dst", "ts"]].rename(columns={"dst": "account"})])
    g = both.groupby("account")["ts"]
    span = ((g.max() - g.min()).dt.total_seconds() / 3600).rename("active_span_hours")

    # Dwell: for each outgoing payment, hours since the account last received money
    sends = df[["src", "ts"]].rename(columns={"src": "account"}).sort_values("ts")
    recvs = df[["dst", "ts"]].rename(columns={"dst": "account", "ts": "ts_in"}).sort_values("ts_in")
    m = pd.merge_asof(sends, recvs, left_on="ts", right_on="ts_in", by="account", direction="backward")
    m["dwell_h"] = (m["ts"] - m["ts_in"]).dt.total_seconds() / 3600
    gd = m.groupby("account")["dwell_h"]
    dwell = pd.DataFrame({
        "dwell_min_h": gd.min(),
        "dwell_median_h": gd.median(),
        "share_out_within_24h_of_in": m.assign(fast=m["dwell_h"] <= 24).groupby("account")["fast"].mean(),
    })

    fmt_in = pd.crosstab(df["dst"], df["payment_format"], normalize="index")
    fmt_in.columns = [f"fmt_in_{str(c).lower().replace(' ', '_')}" for c in fmt_in.columns]
    fmt_in.index.name = "account"

    feats = pd.concat([out, inn, span, dwell, fmt_in], axis=1).reindex(accounts)
    # No incoming money before a send -> no dwell; use a large sentinel instead of 0
    for c in ["dwell_min_h", "dwell_median_h"]:
        feats[c] = feats[c].fillna(1e4)
    return feats.fillna(0.0)


def align_columns(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Payment formats can differ by period; give every split the same columns."""
    cols = sorted(set().union(*(f.columns for f in frames.values())))
    return {k: f.reindex(columns=cols, fill_value=0.0) for k, f in frames.items()}
