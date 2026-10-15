"""Network-analysis features on the account transaction graph.

Edges are aggregated per (sender, receiver) pair first, so the graph stays small
enough for a laptop even with millions of transactions. Cycle and reciprocity
counts use sparse matrix algebra instead of path enumeration.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse


def edge_list(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["src"] != df["dst"]]
        .groupby(["src", "dst"])["amount_usd"]
        .agg(weight="sum", n_tx="size")
        .reset_index()
    )


def graph_features(df: pd.DataFrame, accounts: pd.Index, pagerank: bool = True,
                   louvain: bool = True, seed: int = 42) -> pd.DataFrame:
    edges = edge_list(df)
    nodes = pd.Index(accounts)
    idx = pd.Series(np.arange(len(nodes)), index=nodes)
    edges = edges[edges["src"].isin(nodes) & edges["dst"].isin(nodes)]
    r, c = idx[edges["src"]].to_numpy(), idx[edges["dst"]].to_numpy()
    n = len(nodes)

    A = sparse.csr_matrix((np.ones(len(edges)), (r, c)), shape=(n, n))   # binary adjacency
    W = sparse.csr_matrix((edges["weight"].to_numpy(), (r, c)), shape=(n, n))  # USD-weighted

    feats = pd.DataFrame(index=nodes)
    feats["g_out_degree"] = np.asarray(A.sum(axis=1)).ravel()
    feats["g_in_degree"] = np.asarray(A.sum(axis=0)).ravel()
    feats["g_out_strength"] = np.asarray(W.sum(axis=1)).ravel()
    feats["g_in_strength"] = np.asarray(W.sum(axis=0)).ravel()
    # A->B and B->A both exist: round-tripping between two accounts
    feats["g_reciprocal"] = np.asarray(A.multiply(A.T).sum(axis=1)).ravel()
    # Directed 3-cycles through each node (A->B->C->A): layering loops
    A2 = A @ A
    feats["g_cycles3"] = np.asarray(A2.multiply(A.T).sum(axis=1)).ravel()
    # Distinct accounts reachable in two hops: how far funds can spread
    feats["g_two_hop_reach"] = np.diff((A2 > 0).astype(np.int8).tocsr().indptr)

    if pagerank or louvain:
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(zip(edges["src"], edges["dst"], edges["weight"]))
        if pagerank:
            pr = nx.pagerank(G, weight="weight")
            feats["g_pagerank"] = pd.Series(pr).reindex(nodes).to_numpy()
        if louvain:
            communities = nx.community.louvain_communities(G.to_undirected(), weight="weight", seed=seed)
            size = {}
            for comm in communities:
                for node in comm:
                    size[node] = len(comm)
            feats["g_community_size"] = pd.Series(size).reindex(nodes).fillna(1).to_numpy()
    return feats.fillna(0.0)


def neighbor_aggregates(df: pd.DataFrame, feats: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean of selected features over each account's senders and receivers.

    A one-layer, label-free version of what a graph neural network learns:
    an ordinary-looking account that trades with high-risk-looking accounts
    inherits some of that signal. Labels are never propagated, so no leakage.
    """
    cols = [c for c in cols if c in feats.columns]
    nodes = feats.index
    idx = pd.Series(np.arange(len(nodes)), index=nodes)
    edges = edge_list(df)
    edges = edges[edges["src"].isin(nodes) & edges["dst"].isin(nodes)]
    r, c = idx[edges["src"]].to_numpy(), idx[edges["dst"]].to_numpy()
    n = len(nodes)
    A = sparse.csr_matrix((np.ones(len(edges)), (r, c)), shape=(n, n))
    F = np.log1p(feats[cols].clip(lower=0).to_numpy(dtype=float))

    def mean_over(M):
        deg = np.asarray(M.sum(axis=1)).ravel()
        inv = sparse.diags(np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0))
        return inv @ (M @ F)

    out_mean = mean_over(A)      # over accounts this account pays
    in_mean = mean_over(A.T.tocsr())  # over accounts that pay this account
    res = pd.DataFrame(
        np.hstack([out_mean, in_mean]),
        index=nodes,
        columns=[f"nbr_out_{c}" for c in cols] + [f"nbr_in_{c}" for c in cols],
    )
    return res
