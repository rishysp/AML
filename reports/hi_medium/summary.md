# Key findings (generated from results.json)

- **Data:** 31.9M synthetic bank transactions (IBM AML dataset), split into equal-size chronological windows; results are for the held-out test window.
- **Model vs rules:** at a 1% investigator alert budget, XGBoost caught 87% (95% CI 87%-88%) of laundering accounts at 69% precision, versus 16% recall and 13% precision for tuned rules: 5.4x the detections for the same workload.
- **Network features:** recall at budget 78% without them vs 87% with them (significant in a paired bootstrap).
- **Stability:** PR-AUC SD across seeds 0.0003; a model trained only on the training windows scored PR-AUC 0.61 on validation and 0.90 on test.

Details: `model_comparison_test.csv`, `paired_differences_test.csv`, `bootstrap_ci_test.csv`, `figures/`. A large validation-test gap is a limitation to investigate, not a strength.
