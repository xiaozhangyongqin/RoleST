"""Chronological split dataset and masked forecasting metrics for RoleST."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficWindowDataset(Dataset):
    """Strictly within-partition sliding window dataset over BasicTS memmap data.dat."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        input_len: int = 12,
        output_len: int = 12,
        split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    ) -> None:
        self.data_dir = Path(data_dir)
        desc = json.loads((self.data_dir / "desc.json").read_text(encoding="utf-8"))
        shape = tuple(desc["shape"])
        self._mem = np.memmap(self.data_dir / "data.dat", dtype="float32", mode="r", shape=shape)

        total_t = shape[0]
        t_train = int(round(total_t * split_ratios[0]))
        t_val = int(round(total_t * split_ratios[1]))
        if split == "train":
            self.start, self.stop = 0, t_train
        elif split == "valid":
            self.start, self.stop = t_train, t_train + t_val
        elif split == "test":
            self.start, self.stop = t_train + t_val, total_t
        else:
            raise ValueError(f"Unknown split: {split}")

        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.num_windows = max(0, (self.stop - self.start) - (self.input_len + self.output_len) + 1)

    @property
    def raw_flow(self) -> np.ndarray:
        return np.asarray(self._mem[self.start:self.stop, :, 0], dtype=np.float32)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.start + idx
        x = np.array(self._mem[s : s + self.input_len, :, :3], dtype=np.float32, copy=True)
        y = np.array(self._mem[s + self.input_len : s + self.input_len + self.output_len, :, :1], dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)


def compute_source_stats(train_flow: np.ndarray, null_val: float = 0.0) -> tuple[float, float]:
    valid = np.isfinite(train_flow) & (train_flow != null_val)
    vals = train_flow[valid].astype(np.float64)
    return float(vals.mean()), float(vals.std())


class MaskedMetrics:
    """Exact masked MAE and RMSE over finite non-zero ground-truth entries."""

    def __init__(self, null_val: float = 0.0) -> None:
        self.null_val = null_val
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        mask = torch.isfinite(target) & (target != self.null_val)
        if not mask.any():
            return
        diff = (pred[mask] - target[mask]).double()
        self.abs_sum += float(diff.abs().sum().item())
        self.sq_sum += float(diff.square().sum().item())
        self.count += int(mask.sum().item())

    def compute(self) -> dict[str, float]:
        denom = max(self.count, 1)
        return {
            "MAE": self.abs_sum / denom,
            "RMSE": float(np.sqrt(self.sq_sum / denom)),
        }
