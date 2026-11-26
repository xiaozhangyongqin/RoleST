"""Flow-6 structural role clustering and static source-graph role-relation prior."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans


def load_adjacency(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as f:
        try:
            obj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            obj = pickle.load(f, encoding="latin1")
    adj = obj[2] if isinstance(obj, (list, tuple)) else obj
    return np.asarray(adj, dtype=np.float32)


def fit_flow_roles(train_flow: np.ndarray, num_roles: int = 48, seed: int = 2026, null_val: float = 0.0) -> np.ndarray:
    """Cluster source sensors into P structural roles from six training-split flow statistics.

    Args:
        train_flow: [T_train, N] raw traffic flow array from the source training split only.
    Returns:
        labels: [N] int64 role indices in {0, ..., P-1}.
    """
    n = train_flow.shape[1]
    feats = np.zeros((n, 6), dtype=np.float64)
    for i in range(n):
        col = train_flow[:, i]
        v = col[np.isfinite(col) & (col != null_val)]
        if v.size == 0:
            continue
        mu = float(v.mean())
        sd = float(v.std())
        q10, q50, q90 = np.percentile(v, [10, 50, 90])
        feats[i] = [
            np.log1p(max(mu, 0.0)),
            np.log1p(max(sd, 0.0)),
            np.log1p(max(q10, 0.0)),
            np.log1p(max(q50, 0.0)),
            np.log1p(max(q90, 0.0)),
            q90 / (mu + 1e-6),
        ]
    mean = feats.mean(axis=0, keepdims=True)
    std = np.maximum(feats.std(axis=0, keepdims=True), 1e-6)
    z = (feats - mean) / std
    km = KMeans(n_clusters=num_roles, n_init=20, random_state=seed)
    return km.fit_predict(z).astype(np.int64)


class StaticSourceGraphPrior:
    """Computes the node-agnostic P x P static source-graph role-relation target K_T."""

    def __init__(self, adj: np.ndarray, labels: np.ndarray, num_roles: int, device: torch.device) -> None:
        a = np.asarray(adj, dtype=np.float32).copy()
        np.fill_diagonal(a, 1.0)
        recv, send = np.nonzero(a > 0)
        weights = a[recv, send].astype(np.float32)

        self.num_roles = int(num_roles)
        self.receiver = torch.as_tensor(recv, dtype=torch.long, device=device)
        self.sender = torch.as_tensor(send, dtype=torch.long, device=device)
        self.edge_weight = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.labels = torch.as_tensor(labels, dtype=torch.long, device=device)

    @torch.no_grad()
    def coarse_target(self, available: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool receiver-normalised static source edge weights into K_T [B, P, P]."""
        b, n = available.shape
        p = self.num_roles
        device = available.device

        valid_edge = available[:, self.receiver] & available[:, self.sender]  # [B, E]
        w = self.edge_weight.unsqueeze(0) * valid_edge.to(torch.float32)

        denom = torch.zeros((b, n), dtype=torch.float32, device=device)
        denom.scatter_add_(1, self.receiver.unsqueeze(0).expand(b, -1), w)
        w_norm = w / denom.gather(1, self.receiver.unsqueeze(0).expand(b, -1)).clamp_min(1e-12)

        r_recv = self.labels[self.receiver]
        r_send = self.labels[self.sender]
        pair_idx = (r_recv * p + r_send).unsqueeze(0).expand(b, -1)

        kt_flat = torch.zeros((b, p * p), dtype=torch.float32, device=device)
        kt_flat.scatter_add_(1, pair_idx, w_norm)
        kt = kt_flat.view(b, p, p)

        counts = torch.zeros((b, p), dtype=torch.float32, device=device)
        counts.scatter_add_(1, self.labels.unsqueeze(0).expand(b, -1), available.to(torch.float32))
        active_rows = counts > 0
        kt = kt / counts.clamp_min(1.0).unsqueeze(-1)
        return kt, active_rows
