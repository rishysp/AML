# GNN benchmark: key findings (generated from gnn/results_gnn.json)

- A directional GraphSAGE GNN (PyTorch, GPU) on 31.9M synthetic transactions (HI-Medium) combined with XGBoost raised recall at a 1% alert budget from 87.4% to 88.4% (PR-AUC 0.905 to 0.915), significant in a paired bootstrap.

Recall at 1% budget: XGBoost 87.4%, GNN 88.2%, ensemble 88.4%.
GNN seed spread (PR-AUC sd): 0.0002.

Compare models within one test window only; absolute scores depend on the window's length and base rate.
