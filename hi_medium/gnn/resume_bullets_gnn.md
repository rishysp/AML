# GNN resume bullet drafts (auto-generated from gnn/results_gnn.json)

- Built a directional GraphSAGE graph neural network in PyTorch (GPU) on 31.9M synthetic transactions (HI-Medium); combined with XGBoost, it raised recall at a 1% alert budget from 87% to 88% (PR-AUC 0.90 to 0.91), significant in a paired bootstrap.

Recall at 1% budget: XGBoost 87.4%, GNN 88.2%, ensemble 88.4%.
GNN seed spread (PR-AUC sd): 0.0002.

Only use the 'raised recall' version if the script wrote it: it checks significance for you.
A well-run benchmark that keeps the simpler model is a good interview story too.
