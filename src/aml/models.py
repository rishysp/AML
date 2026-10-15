"""Supervised (XGBoost), unsupervised (Isolation Forest) and rule-baseline scorers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                  params: dict, random_state: int = 42) -> XGBClassifier:
    """Class imbalance handled with scale_pos_weight; early stopping on validation PR-AUC."""
    pos = max(1, int(y_train.sum()))
    model = _clf(
        **{k: v for k, v in params.items() if k != "device"},
        scale_pos_weight=(len(y_train) - pos) / pos,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


DEVICE = "cpu"  # set to "cuda" (from config models.xgboost.device) to train XGBoost on a GPU


def _clf(**kw) -> XGBClassifier:
    return XGBClassifier(device=DEVICE, **kw)


SEARCH_SPACE = {
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 20],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "reg_lambda": [1.0, 5.0, 20.0],
    "gamma": [0.0, 1.0],
    # How strongly to up-weight positives: full imbalance ratio, its square root, or none.
    # Full weighting often hurts ranking quality (PR-AUC) on very rare classes.
    "pos_weight_mode": ["full", "sqrt", "none"],
}


def _pos_weight(y: pd.Series, mode: str) -> float:
    pos = max(1, int(y.sum()))
    ratio = (len(y) - pos) / pos
    return {"full": ratio, "sqrt": float(np.sqrt(ratio)), "none": 1.0}[mode]


def tune_xgboost(X_train, y_train, X_val, y_val, base: dict, n_trials: int, seed: int = 42):
    """Random search scored on validation PR-AUC. Returns (best_params, best_iteration, trials)."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    trials = []
    best = (-1.0, None, None)
    for t in range(n_trials):
        p = {k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()}
        if t == 0:  # always include the v1 settings as a reference point
            p.update({k: base[k] for k in ["max_depth", "learning_rate", "subsample", "colsample_bytree"]},
                     min_child_weight=1, reg_lambda=1.0, gamma=0.0, pos_weight_mode="full")
        mode = p.pop("pos_weight_mode")
        model = _clf(
            n_estimators=base["n_estimators"], early_stopping_rounds=base["early_stopping_rounds"],
            eval_metric="aucpr", tree_method="hist", scale_pos_weight=_pos_weight(y_train, mode),
            random_state=seed, n_jobs=-1, **p,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        score = float(average_precision_score(y_val, model.predict_proba(X_val)[:, 1]))
        trials.append({**p, "pos_weight_mode": mode, "val_pr_auc": score,
                       "best_iteration": int(model.best_iteration)})
        print(f"  trial {t + 1}/{n_trials}: val PR-AUC {score:.4f}  {p} {mode}", flush=True)
        if score > best[0]:
            best = (score, {**p, "pos_weight_mode": mode}, int(model.best_iteration))
    return best[1], best[2], pd.DataFrame(trials).sort_values("val_pr_auc", ascending=False)


def find_rounds(X_train, y_train, X_val, y_val, params: dict, max_rounds: int,
                early_stopping: int, seed: int = 42) -> tuple[int, float]:
    """Best boosting-round count for fixed settings, with a high cap (early stopping on validation)."""
    from sklearn.metrics import average_precision_score

    p = dict(params)
    mode = p.pop("pos_weight_mode")
    model = _clf(n_estimators=max_rounds, early_stopping_rounds=early_stopping,
                          eval_metric="aucpr", tree_method="hist",
                          scale_pos_weight=_pos_weight(y_train, mode), random_state=seed, n_jobs=-1, **p)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    score = float(average_precision_score(y_val, model.predict_proba(X_val)[:, 1]))
    return int(model.best_iteration), score


def fit_final(X, y, params: dict, n_estimators: int, seed: int = 42) -> XGBClassifier:
    """Refit with fixed rounds (no early stopping) once the settings are chosen."""
    p = dict(params)
    mode = p.pop("pos_weight_mode")
    model = _clf(n_estimators=n_estimators, tree_method="hist", eval_metric="aucpr",
                          scale_pos_weight=_pos_weight(y, mode), random_state=seed, n_jobs=-1, **p)
    model.fit(X, y, verbose=False)
    return model


def train_isolation_forest(X_train: pd.DataFrame, params: dict, random_state: int = 42) -> IsolationForest:
    """Unsupervised: never sees labels. Useful where labelled cases are scarce."""
    model = IsolationForest(**params, random_state=random_state, n_jobs=-1)
    model.fit(np.log1p(X_train.clip(lower=0)))
    return model


def isolation_scores(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    return -model.score_samples(np.log1p(X.clip(lower=0)))  # higher = more anomalous


def rule_baseline_score(scenario_scores: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    """Combined rule score: number of scenarios fired, tie-broken by how far past threshold.

    Reproduces what a rules-only monitoring program would prioritise.
    """
    fired = sum((scenario_scores[k] >= t).astype(float) for k, t in thresholds.items())
    margin = sum((scenario_scores[k] / max(t, 1e-9)).clip(upper=10) for k, t in thresholds.items())
    return (fired + margin / (10 * len(thresholds) + 1)).to_numpy()
