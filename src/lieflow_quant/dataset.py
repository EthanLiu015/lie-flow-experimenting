"""PyTorch datasets for equity cross-section point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class CrossSectionCloudDataset(Dataset):
    """
    Load pre-built daily cross-section point clouds for SO(3) symmetry discovery.

    Data file: ``clouds.npy`` with shape (N_days, n_stocks, 3) for features
    (momentum, realized_vol, log_market_cap), cross-sectionally standardized
    per day. Optional ``metadata.npz`` with ``dates``, ``vix``, ``regime``.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        train_ratio: float = 0.8,
        num_samples: Optional[int] = None,
        random_seed: int = 42,
        metadata_path: Optional[str] = None,
        regime: Optional[str] = None,
        split_mode: str = "random",
    ):
        assert split in ("train", "test")
        assert split_mode in ("random", "temporal")
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Cloud data not found: {data_path}")

        raw = np.load(data_path)
        assert raw.ndim == 3 and raw.shape[2] == 3, (
            f"Expected (N, n_stocks, 3), got {raw.shape}"
        )

        meta = None
        if metadata_path and Path(metadata_path).exists():
            meta = np.load(metadata_path, allow_pickle=True)

        indices = np.arange(raw.shape[0])
        if meta is not None and regime is not None and "regime" in meta:
            regime_mask = meta["regime"] == regime
            indices = indices[regime_mask]

        if split_mode == "temporal":
            n_train = int(len(indices) * train_ratio)
            split_indices = indices[:n_train] if split == "train" else indices[n_train:]
        else:
            rng = np.random.default_rng(random_seed)
            perm = rng.permutation(indices)
            n_train = int(len(perm) * train_ratio)
            split_indices = perm[:n_train] if split == "train" else perm[n_train:]

        if num_samples is not None and num_samples < len(split_indices):
            rng = np.random.default_rng(random_seed)
            split_indices = rng.choice(split_indices, size=num_samples, replace=False)

        self.data = torch.from_numpy(raw[split_indices].astype(np.float32))
        self.metadata = None
        if meta is not None:
            self.metadata = {}
            for k in meta.files:
                arr = meta[k]
                if arr.ndim == 0:
                    self.metadata[k] = arr.item()
                elif k == "regime" and regime is not None:
                    self.metadata[k] = np.full(len(split_indices), regime)
                else:
                    self.metadata[k] = arr[split_indices]

        self.split = split
        self.split_mode = split_mode
        self.name = f"cross_section_{split}"
        if regime:
            self.name += f"_{regime}"

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
