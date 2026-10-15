"""Prior case history: exposure to accounts confirmed as laundering in an EARLIER window.

Real monitoring teams know which customers were previously confirmed suspicious
(e.g. filed reports) and treat money flowing to or from them as a risk signal.
Here, "confirmed" means labelled positive in a window that ended at least
`lag` windows before the current one, which mimics investigation delay:
labels from the window being scored, or the one right before it, are never used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HIST_COLS = ["hist_flagged_before", "hist_n_bad_cp", "hist_share_bad_cp", "hist_usd_with_bad"]


def exposure_features(df: pd.DataFrame, accounts: pd.Index, known_bad: set | None) -> pd.DataFrame:
    """Per account: previously flagged?, how many flagged counterparties, share, USD volume with them.

    Returns all-NaN columns when no history is available yet (earliest windows);
    XGBoost treats NaN as "unknown", which is different from "no exposure".
    """
    if known_bad is None:
        return pd.DataFrame(np.nan, index=accounts, columns=HIST_COLS, dtype="float32")

    e = df.loc[df["src"] != df["dst"], ["src", "dst", "amount_usd"]]
    pairs = pd.concat([
        e.rename(columns={"src": "account", "dst": "cp"}),
        e.rename(columns={"dst": "account", "src": "cp"}),
    ], ignore_index=True)
    pairs["bad"] = pairs["cp"].isin(known_bad)
    n_cp = pairs.groupby("account")["cp"].nunique()
    bad = pairs[pairs["bad"]]
    n_bad = bad.groupby("account")["cp"].nunique()
    usd_bad = bad.groupby("account")["amount_usd"].sum()

    out = pd.DataFrame(index=accounts)
    out["hist_flagged_before"] = accounts.isin(known_bad).astype("float32")
    out["hist_n_bad_cp"] = n_bad.reindex(accounts).fillna(0.0)
    out["hist_share_bad_cp"] = (n_bad / n_cp).reindex(accounts).fillna(0.0)
    out["hist_usd_with_bad"] = usd_bad.reindex(accounts).fillna(0.0)
    return out.astype("float32")
