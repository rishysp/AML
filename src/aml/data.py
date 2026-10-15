"""Loading, cleaning, currency normalisation, chronological splitting and account labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# IBM AML CSV column order:
# Timestamp, From Bank, Account, To Bank, Account(.1), Amount Received, Receiving Currency,
# Amount Paid, Payment Currency, Payment Format, Is Laundering
STANDARD_COLUMNS = [
    "ts", "src_bank", "src_acct", "dst_bank", "dst_acct",
    "amount_recv", "curr_recv", "amount_paid", "curr_paid",
    "payment_format", "is_laundering",
]


RAW_DTYPES = {
    "ts": str, "src_bank": "category", "src_acct": str, "dst_bank": "category", "dst_acct": str,
    "amount_recv": "float64", "curr_recv": "category", "amount_paid": "float64",
    "curr_paid": "category", "payment_format": "category", "is_laundering": "int8",
}


def load_raw(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the raw IBM CSV with compact dtypes and stable snake_case names (by position).

    Categories and floats instead of text keep HI-Medium (~32M rows) within a
    Kaggle notebook's memory.
    """
    header = pd.read_csv(path, nrows=0).columns
    if len(header) != len(STANDARD_COLUMNS):
        raise ValueError(f"Expected {len(STANDARD_COLUMNS)} columns, found {len(header)}: {list(header)}")
    return pd.read_csv(path, nrows=nrows, header=0, names=STANDARD_COLUMNS, dtype=RAW_DTYPES)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type conversion and globally unique account IDs (bank + account). Works in place."""
    out = df
    try:  # the IBM files use one fixed format; parsing it directly is much faster than "mixed"
        out["ts"] = pd.to_datetime(out["ts"], format="%Y/%m/%d %H:%M")
    except (ValueError, TypeError):
        out["ts"] = pd.to_datetime(out["ts"], format="mixed")
    # Paired columns must share one category set to be compared (e.g. paid vs received currency)
    from pandas.api.types import union_categoricals
    for a, b in [("curr_paid", "curr_recv"), ("src_bank", "dst_bank")]:
        if isinstance(out[a].dtype, pd.CategoricalDtype) and isinstance(out[b].dtype, pd.CategoricalDtype):
            cats = union_categoricals([out[a], out[b]]).categories
            out[a] = out[a].cat.set_categories(cats)
            out[b] = out[b].cat.set_categories(cats)
    for col in ["amount_recv", "amount_paid"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["is_laundering"] = pd.to_numeric(out["is_laundering"]).astype("int8")
    # Account numbers are only unique within a bank
    out["src"] = out["src_bank"].astype(str) + "_" + out["src_acct"].astype(str)
    out["dst"] = out["dst_bank"].astype(str) + "_" + out["dst_acct"].astype(str)
    out = out.drop(columns=["src_acct", "dst_acct"])  # now inside src/dst; saves memory on HI-Medium
    out = out.dropna(subset=["ts", "amount_paid", "amount_recv"])
    return out.sort_values("ts").reset_index(drop=True)


def implied_rates_to_base(df: pd.DataFrame, base: str = "US Dollar") -> dict[str, float]:
    """Estimate FX rates to the base currency from the data itself.

    Cross-currency rows carry both the paid and received amount, so
    amount_recv / amount_paid is an observed rate. The median per pair is robust
    to outliers. Rates are derived, not hard-coded, which keeps the white paper honest.
    """
    rates = {base: 1.0}
    cross = df[(df["curr_paid"] != df["curr_recv"]) & (df["amount_paid"] > 0)]
    # paid in X, received in base: 1 X = recv/paid base
    to_base = cross[cross["curr_recv"] == base]
    for cur, grp in to_base.groupby("curr_paid", observed=True):
        rates[cur] = float(np.median(grp["amount_recv"] / grp["amount_paid"]))
    # paid in base, received in X: 1 X = paid/recv base
    from_base = cross[(cross["curr_paid"] == base) & (cross["amount_recv"] > 0)]
    for cur, grp in from_base.groupby("curr_recv", observed=True):
        if cur not in rates:
            rates[cur] = float(np.median(grp["amount_paid"] / grp["amount_recv"]))
    return rates


def add_base_amount(df: pd.DataFrame, rates: dict[str, float]) -> pd.DataFrame:
    """Add amount_usd (paid amount in base currency). Unknown currencies become NaN and are dropped."""
    out = df.copy()
    out["amount_usd"] = out["amount_paid"] * out["curr_paid"].astype(str).map(rates).astype(float)
    missing = out["amount_usd"].isna().mean()
    if missing > 0:
        print(f"[data] dropping {missing:.2%} of rows with no derivable FX rate")
    return out.dropna(subset=["amount_usd"]).reset_index(drop=True)


def prepare(raw_csv: str | Path, processed: str | Path, base: str = "US Dollar",
            nrows: int | None = None) -> pd.DataFrame:
    """Full preparation; caches a parquet file so later runs are fast."""
    processed = Path(processed)
    if processed.exists():
        return pd.read_parquet(processed)
    df = clean(load_raw(raw_csv, nrows=nrows))
    rates = implied_rates_to_base(df, base)
    df = add_base_amount(df, rates)
    processed.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed, index=False)
    return df


def chronological_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> dict[str, pd.DataFrame]:
    """Split by timestamp quantiles: train -> val -> test, never shuffled."""
    t1 = df["ts"].quantile(train_frac)
    t2 = df["ts"].quantile(train_frac + val_frac)
    return {
        "train": df[df["ts"] <= t1].reset_index(drop=True),
        "val": df[(df["ts"] > t1) & (df["ts"] <= t2)].reset_index(drop=True),
        "test": df[df["ts"] > t2].reset_index(drop=True),
    }


def equal_windows(df: pd.DataFrame, n_windows: int) -> list[pd.DataFrame]:
    """Cut the timeline into `n_windows` consecutive windows holding equal transaction counts.

    Count-based features and rule scores scale with how much activity a period
    holds. Building every account-period on same-sized windows keeps train,
    validation and test on the same scale, so thresholds and models transfer.
    """
    edges = df["ts"].quantile([i / n_windows for i in range(1, n_windows)]).tolist()
    bounds = [df["ts"].min() - pd.Timedelta(seconds=1)] + edges + [df["ts"].max()]
    return [
        df[(df["ts"] > lo) & (df["ts"] <= hi)].reset_index(drop=True)
        for lo, hi in zip(bounds[:-1], bounds[1:])
    ]


def account_labels(df: pd.DataFrame) -> pd.Series:
    """An account is positive in a period if it sends or receives any laundering transaction.

    Monitoring alerts on accounts (customers), not single transactions, so the
    unit of analysis is the account-period.
    """
    pos = pd.concat([
        df.loc[df["is_laundering"] == 1, "src"],
        df.loc[df["is_laundering"] == 1, "dst"],
    ]).unique()
    accounts = pd.Index(pd.concat([df["src"], df["dst"]]).unique(), name="account")
    return pd.Series(accounts.isin(pos).astype("int8"), index=accounts, name="label")
