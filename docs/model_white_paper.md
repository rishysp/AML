# Model White Paper: AML Transaction Monitoring Prototype

| Item | Detail |
|---|---|
| Model name | AML-TM-Prototype v2 |
| Owner | Rishy Parasar Sridharan |
| Date | October 2026 |
| Status | Prototype on synthetic data. Not approved for production use. |
| Code | `scripts/run_pipeline.py`, `src/aml/`; all parameters in `config.yaml` |
| Evidence | `reports/results.json`, `reports/*.csv`, `reports/figures/` |

## Executive summary

This prototype ranks bank accounts by money-laundering risk so that investigators review the riskiest accounts first, within a fixed review capacity. It combines three rule scenarios (structuring, rapid movement of funds, fan-in/fan-out) with a gradient-boosted model (XGBoost) that uses behavioural, temporal and network features.

On a held-out test period of 356,393 accounts, with investigators reviewing 1% of accounts (3,564 alerts):

| | Laundering accounts caught (recall) | Alerts that are laundering (precision) |
|---|---|---|
| Rules only, tuned | 12.3% | 7.6% |
| **XGBoost model** | **63.2%** (95% CI 61.1-65.1%) | **39.0%** (95% CI 36.9-41.0%) |

The model finds about 5x as many laundering accounts as the rules for the same investigator workload. The gap is statistically significant (paired bootstrap, 200 resamples). Results come from synthetic data and must be revalidated on production data before any use (Section 6).

## 1. Purpose and intended use

- **Business problem.** Identify accounts whose transaction behaviour indicates possible money laundering, so investigators can prioritise reviews.
- **Unit of analysis.** The account-period: one account's activity within one time window. An alert means "review this account's activity in this window".
- **Intended users.** AML investigators and the transaction monitoring team.
- **Decision supported.** Which accounts to review, given a fixed capacity. The design point is an alert budget of 1% of active accounts per window.
- **Out of scope.** Sanctions screening, KYC risk rating, real-time payment blocking, and the decision to file a report. The model prioritises reviews; it does not make determinations.

## 2. Data

### 2.1 Source
IBM "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models", HI-Small variant: an agent-based simulation of banks, accounts and payments with ground-truth laundering labels. No real customer data is used.

| Property | Value |
|---|---|
| Transactions | 5,078,345 |
| Period | 1-18 September 2022 |
| Accounts | 515,088 (unique bank + account number) |
| Banks | 30,528 |
| Currencies | 15 |
| Payment formats | ACH, Bitcoin, Cash, Cheque, Credit Card, Reinvestment, Wire |
| Laundering transactions | 0.10% |
| Laundering accounts (test window) | 2,199 of 356,393 (0.62%) |

### 2.2 Label definition
An account is positive in a window if it sends or receives at least one laundering transaction in that window. This approximates a confirmed case. It differs from production, where labels come from investigation outcomes, arrive weeks later and are incomplete. Section 6 discusses the consequences.

### 2.3 Preparation
- **Account identity.** Account numbers are unique only within a bank, so accounts are keyed on bank plus account number.
- **Currency.** All amounts are converted to USD. Rates are not hard-coded. Each currency's rate is derived from the data's own cross-currency payments (1.4% of transactions), as the median ratio of amount received to amount paid. Every currency received a rate, so no rows were dropped.
- **Ordering.** Transactions are sorted by timestamp and never shuffled.

### 2.4 Data quality checks

| Check | Result | Action |
|---|---|---|
| Missing values | None in any column | None |
| Exact duplicate transactions | 9 of 5,078,345 | Kept: negligible, and genuine repeat payments are possible |
| Zero or negative amounts | None | None |
| Self-transfers (same account) | 591,212 (11.6%) | Kept as activity; excluded from graph edges; see 3.4 on the feature |
| Unconvertible currencies | None | None |

### 2.5 Exploratory findings that shaped the design
- **Payment channel concentrates risk.** ACH has a laundering rate of 0.75% of transactions, about 20x that of any other format, and carries 87% of all laundering transactions (4,483 of 5,177). Wire and reinvestment transactions carry no laundering labels at all. This motivated payment-mix features on both the sending and receiving side.
- **Amounts are heavy-tailed.** Median $894, 99th percentile about $3.1M. Amount features are therefore log-transformed for the unsupervised model, and the tree model is insensitive to scale.
- **Activity is front-loaded.** The first window (equal transaction count) spans 21 hours; the last spans 10 days. This drove the windowing design in Section 3.1.

## 3. Methodology

### 3.1 Time windows
The timeline is cut into five consecutive windows with equal transaction counts (about 1.02M each). Windows 1-3 are training samples, window 4 is validation and window 5 is the test set.

**Why.** Counts such as "number of counterparties" or "payments just under $10,000" scale with how much activity a period contains. In the first version (v1), training covered 60% of transactions and test 20%. Thresholds tuned on training did not transfer: for rapid movement, no threshold met the alert budget on the training data, so the strictest one was used by default. Equal windows put every account-period on the same scale.

### 3.2 Rule scenarios

| Scenario | Typology | Score | Fixed parameters |
|---|---|---|---|
| Structuring | Splitting funds to stay below a reporting threshold | Count of outgoing payments in [$9,000, $10,000) | Reporting threshold $10,000; band 90-100% |
| Rapid movement | Pass-through or mule accounts | Largest 3-day inflow whose outflow is 80-120% of it | 3-day window; ratio band 0.8-1.2 |
| Fan-in / fan-out | Collection and distribution accounts | Maximum of unique senders or unique receivers | None |

Each scenario produces a score per account; an alert fires when the score reaches the scenario's threshold (Section 4).

### 3.3 Machine learning model
- **Algorithm.** XGBoost gradient-boosted trees, chosen because they handle mixed-scale tabular features, missing values and non-linear interactions, and give feature importances that support explanation.
- **Features (77).** All computed per account-window, from transactions in that window only:
  - *Behavioural:* sent and received counts, sums, means, maxima and dispersion; cross-bank and cross-currency shares; round-amount share; near-threshold count; night-time share; payment-format mix.
  - *Temporal:* hourly and daily bursts, activity span, and dwell time (hours between receiving money and sending it on).
  - *Network (31):* degree, USD strength, reciprocity, directed 3-cycles, two-hop reach, PageRank, Louvain community size, plus the average of key features over each account's senders and receivers.
  - *Rule scores:* the three scenario scores.
- **Class imbalance.** Tested explicitly: full up-weighting of positives, square-root weighting and none. No weighting gave the best validation ranking quality, so it was used.
- **Selected settings.** Tree depth 8, learning rate 0.03, minimum child weight 20, row subsample 0.85, column subsample 0.7, L2 penalty 5, 661 boosting rounds.
- **Training procedure.** Settings were chosen by a 10-trial random search scored on validation PR-AUC (0.455 for the v1 settings vs 0.540 for the selected ones). The final model was then refitted on training plus validation windows.

### 3.4 Alternatives considered and rejected

| Alternative | Outcome | Decision |
|---|---|---|
| Isolation Forest (unsupervised) | Test PR-AUC 0.072, recall at budget 18.2% | Kept only as a benchmark. It flags accounts that are unusual, which here is mostly legitimate high activity. |
| Transaction-level model | Not built | Investigators review accounts, not payments. |
| `self_transfer` feature | Severe drift between train and test (PSI 1.38). Removing it moved test PR-AUC from 0.632 to 0.630. | Removed: drift risk with no benefit. |
| Prior-case history features (exposure to accounts confirmed in a window at least 2 windows earlier) | PR-AUC 0.623 with them vs 0.630 without (significant). Recall difference not significant. | Rejected. Only one training window had history, so the model overfit it. Worth retesting with a longer history. |
| Boosting rounds cap raised from 600 to 2,000 | Best count about 600-700; no change in test performance | No change. |
| Graph neural network | Planned on the larger HI-Medium dataset | Pending; see Section 8. |

## 4. Threshold setting

### 4.1 Method
For each scenario, candidate thresholds were swept on the **validation** window, which matches the test window in size. The chosen threshold gives the highest recall whose alert rate stays within the 1% budget. Performance is then reported on the test window, which played no part in the choice.

### 4.2 Chosen thresholds and test performance

| Scenario | Threshold | Test alerts | Alert rate | Precision | Recall |
|---|---|---|---|---|---|
| Fan-in / fan-out | 8 unique counterparties | 2,680 | 0.75% | 7.1% | 8.6% |
| Rapid movement | $10,000 inflow | 2,780 | 0.78% | 6.3% | 8.0% |
| Structuring | 3 near-threshold payments | 586 | 0.16% | 4.4% | 1.2% |

Charts: `reports/figures/threshold_*.png`. Full sweeps: `reports/sweep_val_*.csv` and `reports/sweep_test_*.csv`.

### 4.3 Below-the-line testing
Below-the-line (BTL) testing samples accounts just under each threshold. If they are laundering at a similar rate to alerted accounts, the threshold is too high.

| Scenario | Band just below threshold | Laundering rate in that band | Laundering rate when alerted |
|---|---|---|---|
| Fan-in / fan-out | 5-7 counterparties | 2.3% | 7.1% |
| Rapid movement | $5,000-$9,999 | 2.3% | 6.3% |
| Structuring | 2 payments | 2.2% | 4.4% |

In every scenario, the band just below the line is markedly less productive than the alerted population, so the thresholds are not set too high. This was not true in v1, where the rapid-movement band just below the threshold matched the alerted rate (8.3% vs 8.1%). That finding led to the redesign in Section 3.1.

### 4.4 Interpretation
Individually, the rules have low recall (1-9%), consistent with the high false-positive rates typical of rule-based monitoring. They remain useful as explainable typology coverage and as inputs to the model. Two of the three rule scores, structuring and near-threshold counts, are among the model's top features.

## 5. Performance (held-out test window)

### 5.1 Comparison at the 1% alert budget

| Model | PR-AUC | Recall at budget | Precision at budget | Alerts needed to catch 80% |
|---|---|---|---|---|
| **XGBoost with network features** | **0.630** | **63.2%** | **39.0%** | **14,054 (3.9% of accounts)** |
| XGBoost without network features | 0.517 | 55.0% | 34.0% | 20,542 |
| Isolation Forest (unsupervised) | 0.072 | 18.2% | 11.2% | 104,896 |
| Rules only, tuned | 0.036 | 12.3% | 7.6% | 157,060 |

Base rate (a random ranking): PR-AUC 0.006. Charts: `figures/pr_curves.png` and `figures/alert_efficiency.png`.

**For a non-technical reader.** If investigators can review 3,564 accounts in this period, the model's list contains about 1,390 laundering accounts; the rules' list contains about 270. To catch 80% of laundering accounts, the model needs about 14,000 reviews; the rules would need about 157,000.

### 5.2 Statistical confidence
95% confidence intervals come from 200 account-level bootstrap resamples. Differences are paired, using the same resample for both models.

| Comparison | Metric | Difference | 95% CI | Significant |
|---|---|---|---|---|
| Model vs rules | Recall at budget | +50.8 pts | +48.8 to +53.2 | Yes |
| Network features (with vs without) | Recall at budget | +8.2 pts | +6.6 to +9.8 | Yes |
| Network features (with vs without) | PR-AUC | +0.113 | +0.100 to +0.128 | Yes |

### 5.3 Stability
- **Across random seeds.** Three refits with different seeds: PR-AUC 0.630, standard deviation 0.0008.
- **Across time.** A model trained on the three training windows only scored PR-AUC 0.535 on validation and 0.618 on test. The difference reflects variation between periods rather than a trend, but it means a single period's result can move by about 0.08 PR-AUC. Performance should be judged over several periods.
- **Refit contribution.** The train-only model scores 0.618 on test and the refitted model 0.630, so the gain is not explained by the extra validation data.

### 5.4 What drives the model
Top features by importance:
1. Unique currencies sent
2. Unique currencies received
3. Share of incoming payments by ACH
4. Cross-currency share of an account's senders (network)
5. Share of outgoing payments by ACH
6. Structuring score (rule)
7. Near-threshold payment count
8. Cross-currency share
9. ACH share of an account's senders (network)
10. Peak daily outgoing burst

These match recognised typologies: currency layering, a channel with concentrated risk, structuring, and bursts of activity. No single feature dominates. The full chart is `figures/feature_importance.png`, with network features in green.

## 6. Assumptions and limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Synthetic data | Typologies may be cleaner than real activity, which can overstate performance | Revalidate on production data before any use; expect lower absolute numbers |
| Label = any laundering transaction in the window | Real labels come from investigations and are delayed and incomplete | Train on investigation outcomes in production; test label-lag sensitivity |
| Short history (18 days) | Too little history for long-horizon behaviour or prior-case features (Section 3.4) | Retest on a longer dataset (Section 8) |
| Windows of unequal duration (21 hours to 10 days) | Time-based features such as activity span mean different things across windows | Count-based scaling is controlled; duration-normalised features are a candidate improvement |
| Period-to-period variation (validation 0.535 vs test 0.618) | A single period can mislead | Monitor over multiple periods; set alert-volume triggers (Section 7) |
| Accounts are evaluated within a window | Behaviour spanning a window boundary is split | Overlapping windows are a candidate improvement |
| Fixed 1% alert budget | Capacity varies in practice | Precision and recall curves are provided for other budgets |

## 7. Ongoing monitoring

| What | How | Trigger for action |
|---|---|---|
| Alert volume | Alerts per window vs budget | Moves more than 25% from expected for two consecutive periods |
| Alert productivity | Share of alerts escalated after review | Drops below the agreed floor (start with the validated precision of about 35%) |
| Feature drift | PSI per feature, each period | PSI above 0.25 for any top-20 feature (currently all at or below 0.18) |
| Score drift | PSI of the model score | PSI above 0.25 |
| Below-the-line productivity | Sample just below each rule threshold | Laundering rate in the first band below reaches 75% of the alerted rate |
| Full revalidation | Independent review | Annually, or after any trigger |

At validation, the largest feature drift was moderate: maximum received amount (PSI 0.18) and hourly or daily bursts (0.11-0.14). Everything else was stable. `reports/feature_drift_psi.csv` is the baseline for future monitoring.

## 8. Planned work
- **Graph neural network.** GraphSAGE on the account graph, tested on IBM's larger HI-Medium dataset (about 32M transactions). It will be compared with this model on the same windows and metrics, with a paired bootstrap. It will be adopted only if it is significantly better.
- **Prior-case features** retested on the longer HI-Medium history.
- **Duration-normalised and overlapping windows** (Section 6).

## 9. Governance and reproducibility
- **Reproducible.** Fixed random seeds, version-pinned requirements, and every threshold, grid and parameter in `config.yaml`. One command, `python scripts/run_pipeline.py`, regenerates every number in this paper.
- **Tested.** Unit and end-to-end tests run on synthetic data in a few seconds (`pytest`).
- **Explainable.** Every rule maps to a named typology, and every feature to a behaviour an investigator would recognise.
- **For an independent validator.** The repository, this paper and `reports/` are sufficient to rerun and challenge each claim. The version history (`reports/version_history.csv`) and the rejected alternatives (Section 3.4) are part of the record.
