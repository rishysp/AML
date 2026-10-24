# Key findings (generated from results.json)

- **Data:** 5.1M synthetic bank transactions (IBM AML dataset), split into equal-size chronological windows; results are for the held-out test window.
- **Model vs rules:** at a 1% investigator alert budget, XGBoost caught 63% (95% CI 61%-65%) of laundering accounts at 39% precision, versus 12% recall and 8% precision for tuned rules: 5.1x the detections for the same workload.
- **Network features:** recall at budget 55% without them vs 63% with them (significant in a paired bootstrap).
- **Stability:** PR-AUC SD across seeds 0.0009; a model trained only on the training windows scored PR-AUC 0.53 on validation and 0.62 on test.

Details: `model_comparison_test.csv`, `paired_differences_test.csv`, `bootstrap_ci_test.csv`, `figures/`. A large validation-test gap is a limitation to investigate, not a strength.
