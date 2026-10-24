# Resume bullet drafts (auto-generated from reports/results.json)

Pick 3-4. Every number comes from your own held-out test window.

- Built an account-level AML transaction monitoring pipeline in Python on 31.9M synthetic bank transactions (IBM AML dataset), using equal-size chronological windows so thresholds and models transfer between periods.
- At a 1% investigator alert budget, the tuned XGBoost model caught 87% (95% CI 87%-88%) of laundering accounts at 69% precision, versus 16% recall and 13% precision for tuned rules: 5.4x the detections for the same workload.
- Engineered network features with NetworkX and sparse matrices (Louvain communities, PageRank, reciprocity, 3-cycles, counterparty aggregates); recall at budget 78% without them vs 87% with them (significant in a paired bootstrap).
- Validated durability with 95% bootstrap CIs, seed reruns (PR-AUC sd 0.000), validation-vs-test stability (PR-AUC 0.61 vs 0.90) and PSI feature-drift monitoring.
- Implemented structuring, rapid-movement and fan-in/fan-out scenarios with thresholds tuned against the alert budget and checked with below-the-line testing.

## Sanity checks

- Say 'synthetic' when describing the data.
- If the validation and test PR-AUC differ a lot, mention it as a limitation, not a strength.
