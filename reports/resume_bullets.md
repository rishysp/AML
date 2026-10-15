# Resume bullet drafts (auto-generated from reports/results.json)

Pick 3-4. Every number comes from your own held-out test window.

- Built an account-level AML transaction monitoring pipeline in Python on 5.1M synthetic bank transactions (IBM AML dataset), using equal-size chronological windows so thresholds and models transfer between periods.
- At a 1% investigator alert budget, the tuned XGBoost model caught 63% (95% CI 61%-65%) of laundering accounts at 39% precision, versus 12% recall and 8% precision for tuned rules: 5.1x the detections for the same workload.
- Engineered network features with NetworkX and sparse matrices (Louvain communities, PageRank, reciprocity, 3-cycles, counterparty aggregates); recall at budget 55% without them vs 63% with them (significant in a paired bootstrap).
- Validated durability with 95% bootstrap CIs, seed reruns (PR-AUC sd 0.001), validation-vs-test stability (PR-AUC 0.53 vs 0.62) and PSI feature-drift monitoring.
- Implemented structuring, rapid-movement and fan-in/fan-out scenarios with thresholds tuned against the alert budget and checked with below-the-line testing.

## Sanity checks

- Say 'synthetic' when describing the data.
- If the validation and test PR-AUC differ a lot, mention it as a limitation, not a strength.
