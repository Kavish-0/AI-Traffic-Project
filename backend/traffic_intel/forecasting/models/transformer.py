"""Lightweight temporal Transformer forecaster (encoder-only).

    x [N, L, F] -> linear projection to d_model + sinusoidal positions
               -> TransformerEncoder (pre-norm, ``layers`` x multi-head self-attention)
               -> [last-step token ; mean over time] -> linear head -> all horizons x targets

Defaults (d_model 64, 4 heads, 2 layers, ~70k parameters) keep it in the same
size class as the LSTM, so the comparison is about the architecture (attention
over the whole window vs recurrence), not about model size.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from traffic_intel.forecasting.interfaces import FORECASTERS
from traffic_intel.forecasting.models._torch import SequenceForecaster


class _PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class _TransformerNet(nn.Module):
    def __init__(self, n_features: int, d_model: int, heads: int, layers: int, ff: int, dropout: float,
                 n_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(2 * d_model)
        self.head = nn.Linear(2 * d_model, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.pos(self.proj(x)))
        y: torch.Tensor = self.head(self.norm(torch.cat([h[:, -1], h.mean(dim=1)], dim=-1)))
        return y


@FORECASTERS.register("transformer")
class TransformerForecaster(SequenceForecaster):
    name = "transformer"

    def __init__(self, d_model: int = 64, heads: int = 4, layers: int = 2, ff: int = 128, dropout: float = 0.1,
                 **params: Any) -> None:
        super().__init__(d_model=d_model, heads=heads, layers=layers, ff=ff, dropout=dropout, **params)

    def _build_net(self, n_features: int, n_out: int) -> nn.Module:
        p = self.params
        return _TransformerNet(n_features, p["d_model"], p["heads"], p["layers"], p["ff"], p["dropout"], n_out)
