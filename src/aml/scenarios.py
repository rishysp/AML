"""Rule-based monitoring scenarios.

Each scenario returns one numeric *score* per account. An alert fires when the
score is at or above the scenario's threshold, so threshold tuning (tuning.py)
is independent of how the score is computed.
"""
from __future__ import annotations

import pandas as pd


def structuring_score(df: pd.DataFrame, reporting_threshold: float, band_lower_frac: float) -> pd.Series:
    """Count of outgoing transactions just below the reporting threshold.

    Typology: splitting funds into amounts that stay under a reporting limit.
    """
    lower = reporting_threshold * band_lower_frac
    in_band = df[(df["amount_usd"] >= lower) & (df["amount_usd"] < reporting_threshold)]
    return in_band.groupby("src").size().rename("structuring")


def rapid_movement_score(df: pd.DataFrame, window_days: int,
                         ratio_min: float, ratio_max: float) -> pd.Series:
    """Largest inflow that left the account again within `window_days`.

    Typology: pass-through / mule accounts where money comes in and goes out
    quickly in roughly the same amount. Score = max rolling inflow over windows
    whose outflow/inflow ratio is inside [ratio_min, ratio_max]; 0 otherwise.
    """
    day = df["ts"].dt.floor("D")
    inflow = df.assign(day=day).groupby(["dst", "day"])["amount_usd"].sum().rename("inflow")
    outflow = df.assign(day=day).groupby(["src", "day"])["amount_usd"].sum().rename("outflow")
    inflow.index.names = ["account", "day"]
    outflow.index.names = ["account", "day"]
    daily = pd.concat([inflow, outflow], axis=1).fillna(0.0).reset_index()
    daily = daily.sort_values(["account", "day"]).set_index("day")

    window = f"{window_days}D"
    rolled = (
        daily.groupby("account")[["inflow", "outflow"]]
        .rolling(window).sum()
        .reset_index()
    )
    ratio = rolled["outflow"] / rolled["inflow"].where(rolled["inflow"] > 0)
    qualifying = rolled["inflow"].where(ratio.between(ratio_min, ratio_max), 0.0)
    return qualifying.groupby(rolled["account"]).max().rename("rapid_movement")


def fan_score(df: pd.DataFrame) -> pd.Series:
    """Max of unique counterparties sent to (fan-out) and received from (fan-in).

    Typology: one account collecting from or distributing to many others
    (smurfing, collection accounts).
    """
    fan_out = df.groupby("src")["dst"].nunique()
    fan_in = df.groupby("dst")["src"].nunique()
    both = pd.concat([fan_out.rename("out"), fan_in.rename("in")], axis=1).fillna(0)
    return both.max(axis=1).rename("fan")


def all_scenario_scores(df: pd.DataFrame, cfg: dict, accounts: pd.Index) -> pd.DataFrame:
    """Score every account on every scenario (0 where a scenario never fires)."""
    s = cfg["scenarios"]
    scores = pd.concat([
        structuring_score(df, s["structuring"]["reporting_threshold"], s["structuring"]["band_lower_frac"]),
        rapid_movement_score(df, s["rapid_movement"]["window_days"],
                             s["rapid_movement"]["passthrough_ratio_min"],
                             s["rapid_movement"]["passthrough_ratio_max"]),
        fan_score(df),
    ], axis=1)
    return scores.reindex(accounts).fillna(0.0)
