"""Recurrent sequence forecasters: LSTM and GRU.

Input : the previous ``lookback_steps`` traffic states (all dynamic features +
        calendar/weather context), shape [N, L, F].
Output: all horizons x targets at once (direct multi-horizon), from the final
        hidden state through a linear head.

LSTM vs GRU: GRU has ~25% fewer parameters (no separate cell state) and often
matches LSTM on short sequences; it is included to test whether the extra
LSTM gating is worth it here.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from traffic_intel.forecasting.interfaces import FORECASTERS
from traffic_intel.forecasting.models._torch import SequenceForecaster


class _RNNNet(nn.Module):
    def __init__(self, cell: str, n_features: int, hidden: int, layers: int, dropout: float, n_out: int) -> None:
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_features, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        y: torch.Tensor = self.head(out[:, -1])
        return y


class _RecurrentForecaster(SequenceForecaster):
    cell = "lstm"

    def __init__(self, hidden: int = 64, layers: int = 2, dropout: float = 0.1, **params: Any) -> None:
        super().__init__(hidden=hidden, layers=layers, dropout=dropout, **params)

    def _build_net(self, n_features: int, n_out: int) -> nn.Module:
        p = self.params
        return _RNNNet(self.cell, n_features, p["hidden"], p["layers"], p["dropout"], n_out)


@FORECASTERS.register("lstm")
class LSTMForecaster(_RecurrentForecaster):
    name = "lstm"
    cell = "lstm"


@FORECASTERS.register("gru")
class GRUForecaster(_RecurrentForecaster):
    name = "gru"
    cell = "gru"
