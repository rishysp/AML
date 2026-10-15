# %% [markdown]
# # Exploratory data analysis: IBM AML (HI-Small)
# Cell-based script: run cell by cell in VS Code (Jupyter extension) or convert with jupytext.
# Save the charts you like to reports/figures/ for the README and white paper.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path.cwd() if (Path.cwd() / "config.yaml").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
from aml import data  # noqa: E402

cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
df = data.prepare(ROOT / cfg["data"]["raw_csv"], ROOT / cfg["data"]["processed"], cfg["data"]["base_currency"])
df.head()

# %% Class balance: expect a heavily imbalanced target
print(f"{len(df):,} transactions, laundering rate {df['is_laundering'].mean():.4%}")
labels = data.account_labels(df)
print(f"{len(labels):,} accounts, laundering-account rate {labels.mean():.4%}")

# %% Laundering rate by payment format: which channels carry the risk?
by_fmt = df.groupby("payment_format")["is_laundering"].agg(["mean", "size"]).sort_values("mean")
ax = by_fmt["mean"].plot.barh(figsize=(6, 4), title="Laundering rate by payment format")
ax.set_xlabel("Share of transactions labelled laundering")
plt.tight_layout()

# %% Amount distributions: laundering vs normal (log scale)
fig, ax = plt.subplots(figsize=(7, 4))
for lab, name in [(0, "Normal"), (1, "Laundering")]:
    ax.hist(df.loc[df["is_laundering"] == lab, "amount_usd"].clip(lower=1),
            bins=pd.Series(range(0, 9)).pipe(lambda s: 10.0 ** s.to_numpy()),
            alpha=0.6, density=True, label=name)
ax.set_xscale("log")
ax.set_xlabel("Amount (USD, log)")
ax.legend()
ax.set_title("Transaction amounts")
plt.tight_layout()

# %% Activity over time: check the chronological split is sensible
daily = df.set_index("ts").resample("D")["is_laundering"].agg(["size", "sum"])
daily.plot(subplots=True, figsize=(8, 5), title=["Transactions per day", "Laundering txns per day"])
plt.tight_layout()

# %% Near-threshold amounts: is there visible structuring?
near = df[(df["amount_usd"] > 8000) & (df["amount_usd"] < 12000)]
near.groupby(pd.cut(near["amount_usd"], 16))["is_laundering"].mean().plot.bar(
    figsize=(8, 3.5), title="Laundering rate by amount band around 10,000")
plt.tight_layout()

# %% Notes for the white paper
# - Write 3-5 observations here; they become Section 2 (Data) and justify your scenarios.
