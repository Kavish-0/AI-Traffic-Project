"""Shared PyTorch training loop for sequence forecasters.

Input windows are gathered on the device from one concatenated array (no
per-sample Python work), so a few hundred thousand windows train quickly.
Loss: MSE on scaled targets (delta or level). Early stopping on the
validation period restores the best weights.

Note on shuffling: the train/val/test split is chronological (windows.py).
Mini-batches are drawn in random order *within the training period*, which is
standard SGD practice and cannot leak future information - every window still
only contains data from before its own targets.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from traffic_intel.forecasting.data.windows import WindowSet
from traffic_intel.forecasting.interfaces import LearnedForecaster

log = logging.getLogger(__name__)


def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested for training but unavailable - using CPU")
        return torch.device("cpu")
    return torch.device(device)


class _Gatherer:
    """Windows [B, L, F] from X_all via index arithmetic, on the device."""

    def __init__(self, ws: WindowSet, lookback: int, device: torch.device) -> None:
        X_all, g = ws.concat()
        self.X = torch.from_numpy(X_all).to(device)
        self.g = torch.from_numpy(g).to(device)
        self.ar = torch.arange(-lookback + 1, 1, device=device)

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.X[self.g[idx][:, None] + self.ar[None, :]]


class SequenceForecaster(LearnedForecaster):
    """Base for LSTM / GRU / Transformer: subclasses build ``self._build_net(F, out)``."""

    family = "deep"
    framework = "torch"

    def __init__(self, lookback_steps: int = 30, batch_size: int = 512, max_epochs: int = 40, lr: float = 1e-3,
                 weight_decay: float = 1e-5, patience: int = 5, grad_clip: float = 1.0, device: str = "auto",
                 seed: int = 0, **params: Any) -> None:
        super().__init__(lookback_steps=lookback_steps, batch_size=batch_size, max_epochs=max_epochs, lr=lr,
                         weight_decay=weight_decay, patience=patience, grad_clip=grad_clip, device=device, seed=seed,
                         **params)
        self.net: nn.Module | None = None
        self.n_features: int | None = None
        self.n_out: int | None = None

    def _build_net(self, n_features: int, n_out: int) -> nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters()) if self.net is not None else 0

    # ------------------------------------------------------------------ #
    def fit(self, train: WindowSet, valid: WindowSet | None = None) -> dict[str, Any]:
        p = self.params
        torch.manual_seed(p["seed"])
        np.random.seed(p["seed"])
        dev = pick_device(p["device"])
        L = int(p["lookback_steps"])
        if train.lookback < L:
            raise ValueError(f"model lookback {L} > window lookback {train.lookback}")
        y_tr = torch.from_numpy(self._targets(train)).to(dev)
        self.n_features, self.n_out = train.n_features, y_tr.shape[1]
        self.net = self._build_net(self.n_features, self.n_out).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, p["patience"] // 2))
        loss_fn = nn.MSELoss()
        gat_tr = _Gatherer(train, L, dev)
        if valid is not None and len(valid):
            gat_va = _Gatherer(valid, L, dev)
            y_va = torch.from_numpy(self._targets(valid)).to(dev)
        best, best_state, bad, history = float("inf"), None, 0, []
        t0 = time.perf_counter()
        gen = torch.Generator(device="cpu").manual_seed(p["seed"])
        for epoch in range(p["max_epochs"]):
            self.net.train()
            perm = torch.randperm(len(train), generator=gen).to(dev)
            tot = 0.0
            for i in range(0, len(perm), p["batch_size"]):
                idx = perm[i:i + p["batch_size"]]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.net(gat_tr(idx)), y_tr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), p["grad_clip"])
                opt.step()
                tot += loss.item() * len(idx)
            tr_loss = tot / len(train)
            va_loss = self._eval_loss(gat_va, y_va) if valid is not None and len(valid) else tr_loss
            sched.step(va_loss)
            history.append({"epoch": epoch + 1, "train_mse": round(tr_loss, 5), "val_mse": round(va_loss, 5)})
            if va_loss < best - 1e-5:
                best, best_state, bad = va_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return {"fit_s": round(time.perf_counter() - t0, 2), "epochs": len(history), "best_val_mse": round(best, 5),
                "device": str(dev), "n_parameters": self.n_parameters(), "n_train": len(train), "history": history}

    @torch.no_grad()
    def _eval_loss(self, gat: _Gatherer, y: torch.Tensor) -> float:
        self.net.eval()  # type: ignore[union-attr]
        tot, bs = 0.0, self.params["batch_size"] * 4
        for i in range(0, len(y), bs):
            idx = torch.arange(i, min(i + bs, len(y)), device=y.device)
            tot += nn.functional.mse_loss(self.net(gat(idx)), y[idx], reduction="sum").item()  # type: ignore[misc]
        return tot / (len(y) * y.shape[1])

    @torch.no_grad()
    def predict(self, data: WindowSet) -> np.ndarray:
        assert self.net is not None
        dev = next(self.net.parameters()).device
        self.net.eval()
        gat = _Gatherer(data, int(self.params["lookback_steps"]), dev)
        outs, bs = [], self.params["batch_size"] * 4
        for i in range(0, len(data), bs):
            idx = torch.arange(i, min(i + bs, len(data)), device=dev)
            outs.append(self.net(gat(idx)).float().cpu().numpy())
        flat = np.concatenate(outs) if outs else np.zeros((0, self.n_out or 0), np.float32)
        return self._levels(flat, data)

    # ------------------------------------------------------------------ #
    def _save_artifact(self, directory: Path) -> None:
        assert self.net is not None
        torch.save({"state_dict": self.net.state_dict(), "n_features": self.n_features, "n_out": self.n_out},
                   directory / "model.pt")

    def _load_artifact(self, directory: Path) -> None:
        ckpt = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        self.n_features, self.n_out = ckpt["n_features"], ckpt["n_out"]
        self.net = self._build_net(self.n_features, self.n_out)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(pick_device(self.params["device"])).eval()
