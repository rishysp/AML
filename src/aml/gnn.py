"""Directional GraphSAGE for account-level laundering detection (plain PyTorch, no PyG).

Each layer combines an account's own representation with the mean over the
accounts it pays (out-neighbours) and the mean over accounts that pay it
(in-neighbours), each with its own weights:

    h' = act( W_self h + W_out mean_out(h) + W_in mean_in(h) )

Keeping direction separate matters for AML: collecting from many accounts
(fan-in) and distributing to many (fan-out) are different typologies.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

warnings.filterwarnings("ignore", message="Sparse invariant checks")


def transform_features(X: pd.DataFrame) -> np.ndarray:
    """Signed log, so heavy-tailed amounts and counts are on comparable scales. NaN -> 0."""
    v = X.to_numpy(dtype=np.float32, copy=True)
    v = np.sign(v) * np.log1p(np.abs(v))
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


class RunningScaler:
    """Mean and SD accumulated one window at a time (float64), so all training windows
    never need to sit in memory together."""

    def __init__(self):
        self.n, self.s, self.ss = 0, None, None

    def update(self, a: np.ndarray) -> None:
        a = a.astype(np.float64)
        self.s = a.sum(axis=0) if self.s is None else self.s + a.sum(axis=0)
        self.ss = (a * a).sum(axis=0) if self.ss is None else self.ss + (a * a).sum(axis=0)
        self.n += len(a)

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        mean = self.s / self.n
        std = np.sqrt(np.maximum(self.ss / self.n - mean ** 2, 0.0))
        return mean.astype(np.float32), np.where(std > 1e-6, std, 1.0).astype(np.float32)


def fit_scaler(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    sc = RunningScaler()
    for a in arrays:
        sc.update(a)
    return sc.result()


def to_device(g: dict, device) -> dict:
    """Move one window's tensors to `device`. Graphs live in system RAM and are moved one at a
    time, so GPU memory holds a single window however many windows there are."""
    if g["x"].device == torch.device(device):
        return g
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in g.items()}


def _row_norm(rows: np.ndarray, cols: np.ndarray, n: int, device) -> torch.Tensor:
    """Sparse matrix whose product with H gives the mean of H over each row's neighbours."""
    if len(rows) == 0:
        return torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), (n, n),
                                       device=device).coalesce()
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    vals = 1.0 / deg[rows]
    idx = torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(vals), (n, n), device=device).coalesce()


def build_graph(X: pd.DataFrame, labels: pd.Series, edges: pd.DataFrame, mean: np.ndarray,
                std: np.ndarray, device) -> dict:
    """Tensors for one window: features, two neighbour-mean operators, labels."""
    nodes = X.index
    pos = pd.Series(np.arange(len(nodes)), index=nodes)
    e = edges[edges["src"].isin(nodes) & edges["dst"].isin(nodes)]
    s, d = pos[e["src"]].to_numpy(), pos[e["dst"]].to_numpy()
    n = len(nodes)
    # Clip: a column that is nearly constant in training can produce values hundreds of SDs out
    # at test time (drift), which would swamp the network. Trees are immune; neural nets are not.
    x = np.clip((transform_features(X) - mean) / std, -10.0, 10.0).astype(np.float32)
    return {
        "x": torch.from_numpy(x).to(device),
        "A_out": _row_norm(s, d, n, device),  # row i: mean over accounts i pays
        "A_in": _row_norm(d, s, n, device),   # row i: mean over accounts that pay i
        "y": torch.from_numpy(labels.reindex(nodes).to_numpy(dtype=np.float32)).to(device),
        "index": nodes,
    }


class DirSAGELayer(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.self_ = nn.Linear(d_in, d_out)
        self.out_ = nn.Linear(d_in, d_out, bias=False)
        self.in_ = nn.Linear(d_in, d_out, bias=False)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h, A_out, A_in):
        z = self.self_(h) + self.out_(torch.sparse.mm(A_out, h)) + self.in_(torch.sparse.mm(A_in, h))
        return self.norm(z)


class DirSAGE(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        dims = [d_in] + [hidden] * layers
        self.layers = nn.ModuleList(DirSAGELayer(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.drop = nn.Dropout(dropout)
        # The head also sees the raw input, so the GNN can never do worse than "features only" by design
        self.head = nn.Sequential(nn.Linear(hidden + d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, g: dict) -> torch.Tensor:
        h = g["x"]
        for layer in self.layers:
            h = self.drop(torch.relu(layer(h, g["A_out"], g["A_in"])))
        return self.head(torch.cat([h, g["x"]], dim=1)).squeeze(-1)


def pos_weight(graphs: list[dict], mode: str) -> float:
    y = torch.cat([g["y"] for g in graphs])
    p = max(1.0, float(y.sum()))
    ratio = (len(y) - p) / p
    return {"full": ratio, "sqrt": float(np.sqrt(ratio)), "none": 1.0}[mode]


@torch.no_grad()
def predict(model: DirSAGE, g: dict, device=None) -> np.ndarray:
    """Raw logits. Ranking metrics only need the order, and a float32 sigmoid saturates
    to exactly 0 for very low-risk accounts, which creates ties that distort the metrics."""
    model.eval()
    dev = device or next(model.parameters()).device
    return model(to_device(g, dev)).double().cpu().numpy()


def train(train_graphs: list[dict], cfg: dict, seed: int, val_graph: dict | None = None,
          epochs: int | None = None, log=print) -> tuple[DirSAGE, int, list[dict]]:
    """Full-batch training, one window graph per step.

    With `val_graph`: early stopping on validation PR-AUC, returns the best epoch.
    With `epochs`: trains exactly that many epochs (used to refit on train + validation).
    """
    from sklearn.metrics import average_precision_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(cfg.get("device", "cpu"))
    model = DirSAGE(train_graphs[0]["x"].shape[1], cfg["hidden"], cfg["layers"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight(train_graphs, cfg["pos_weight_mode"]),
                                                             device=device))
    max_epochs = epochs or cfg["max_epochs"]
    best_score, best_epoch, best_state, history, since_best = -1.0, 0, None, [], 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for i in rng.permutation(len(train_graphs)):
            g = to_device(train_graphs[i], device)
            opt.zero_grad()
            loss = loss_fn(model(g), g["y"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        row = {"epoch": epoch, "train_loss": total / len(train_graphs)}
        if val_graph is not None:
            score = float(average_precision_score(val_graph["y"].cpu().numpy(), predict(model, val_graph, device)))
            row["val_pr_auc"] = score
            if score > best_score:
                best_score, best_epoch, since_best = score, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1
            if epoch % 10 == 0 or since_best == 0:
                log(f"  epoch {epoch}: loss {row['train_loss']:.4f}, val PR-AUC {score:.4f}"
                    + ("  (best)" if since_best == 0 else ""))
            if since_best >= cfg["patience"]:
                log(f"  early stop at epoch {epoch}; best epoch {best_epoch} (val PR-AUC {best_score:.4f})")
                break
        history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_epoch or max_epochs), history
