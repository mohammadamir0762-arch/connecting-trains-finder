"""Deep learning backend: a neural network with learned embeddings, on the same task as the trees.

Every train, station, zone, train type and weekday gets a learned embedding
(a short vector the network tunes, so trains that behave alike end up close
together). These join the numeric features (route position, rain, festival,
fog, delay history) and pass through two hidden layers that output the
probability of each delay range. It is trained with cross-entropy on the
same halts as the trees and stopped on the same validation weeks, so the two
are compared fairly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from .delay_model import CATEGORICAL, FEATURES, N_BUCKETS

NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
EMBEDDINGS = {"train": 16, "station": 16, "zone": 4, "train_type": 4, "weekday": 3}
UNKNOWN_DROPOUT = 0.05  # train/station ids hidden this often, so "unknown" is learned too


class _Net(nn.Module):
    def __init__(self, n_numeric: int, vocab_sizes: dict[str, int], hidden=(256, 128), dropout: float = 0.2):
        super().__init__()
        self.columns = list(EMBEDDINGS)
        self.embeddings = nn.ModuleDict(
            {f"{c}_ids": nn.Embedding(vocab_sizes[c] + 1, EMBEDDINGS[c]) for c in self.columns})  # 0 = unknown
        width = n_numeric + sum(EMBEDDINGS.values())
        layers = []
        for size in hidden:
            layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, N_BUCKETS))
        self.mlp = nn.Sequential(*layers)

    def forward(self, numeric: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        embedded = [self.embeddings[f"{c}_ids"](ids[:, i]) for i, c in enumerate(self.columns)]
        return self.mlp(torch.cat([numeric, *embedded], dim=1))


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class DeepBackend:
    epochs: int = 10
    batch_size: int = 4096
    learning_rate: float = 2e-3
    patience: int = 2
    vocab: dict[str, dict] = field(default_factory=dict)
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    net: _Net | None = None
    validation_losses: list[float] = field(default_factory=list)

    def _numeric(self, X: pd.DataFrame) -> np.ndarray:
        values = X[NUMERIC].to_numpy(np.float32)
        missing = np.isnan(values)
        values = (np.where(missing, self.mean, values) - self.mean) / self.scale
        return np.hstack([values, missing.astype(np.float32)]).astype(np.float32)

    def _ids(self, X: pd.DataFrame) -> np.ndarray:
        return np.stack([X[c].map(self.vocab[c]).fillna(0).to_numpy(np.int64) for c in EMBEDDINGS], axis=1)

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, seed: int = 0):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.vocab = {c: {v: i + 1 for i, v in enumerate(pd.unique(X[c].dropna()))} for c in EMBEDDINGS}
        values = X[NUMERIC].to_numpy(np.float32)
        self.mean = np.nan_to_num(np.nanmean(values, axis=0)).astype(np.float32)
        self.scale = np.nan_to_num(np.nanstd(values, axis=0), nan=1.0).astype(np.float32)
        self.scale[self.scale == 0] = 1.0

        device = _device()
        numeric, ids = torch.from_numpy(self._numeric(X)), torch.from_numpy(self._ids(X))
        target = torch.from_numpy(np.asarray(y, dtype=np.int64))
        val = (torch.from_numpy(self._numeric(X_val)), torch.from_numpy(self._ids(X_val)),
               torch.from_numpy(np.asarray(y_val, dtype=np.int64)))
        net = _Net(numeric.shape[1], {c: len(v) for c, v in self.vocab.items()}).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        loss_fn = nn.CrossEntropyLoss()

        best, best_state, waited = np.inf, None, 0
        for epoch in range(self.epochs):
            net.train()
            order = torch.from_numpy(rng.permutation(len(target)))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                batch_ids = ids[batch].clone()
                hide = torch.from_numpy(rng.random((len(batch), 2)) < UNKNOWN_DROPOUT)
                batch_ids[:, :2][hide] = 0  # columns 0, 1 are train and station
                optimizer.zero_grad()
                loss = loss_fn(net(numeric[batch].to(device), batch_ids.to(device)), target[batch].to(device))
                loss.backward()
                optimizer.step()
            schedule.step()

            loss = self._loss(net, val, device, loss_fn)
            self.validation_losses.append(loss)
            print(f"    epoch {epoch + 1}: validation log loss {loss:.4f}", flush=True)
            if loss < best - 1e-4:
                best, waited = loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net = net.cpu().eval()
        return self

    def _loss(self, net, data, device, loss_fn) -> float:
        net.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(data[2]), 65536):
                part = [d[start:start + 65536] for d in data]
                total += loss_fn(net(part[0].to(device), part[1].to(device)), part[2].to(device)).item() * len(part[2])
                count += len(part[2])
        return total / count

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not len(X):
            return np.zeros((0, N_BUCKETS))
        numeric, ids = torch.from_numpy(self._numeric(X)), torch.from_numpy(self._ids(X))
        out = []
        with torch.no_grad():
            for start in range(0, len(ids), 65536):
                logits = self.net(numeric[start:start + 65536], ids[start:start + 65536])
                out.append(torch.softmax(logits, dim=1).numpy())
        probs = np.concatenate(out).astype(np.float64)
        return probs / probs.sum(axis=1, keepdims=True)
