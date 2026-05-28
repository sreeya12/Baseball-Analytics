"""
network.py — PyTorch neural network for strike-% regression.

Architecture (with residual connections for 63-file dataset)
─────────────────────────────────────────────────────────────
  Input(36)
    → BatchNorm
    → Linear(256) → BN → ReLU → Dropout
    ↓
    → Linear(128) → BN → ReLU → Dropout  ← residual from 256 (projected)
    ↓
    → Linear(64)  → BN → ReLU → Dropout  ← residual from 128 (projected)
    ↓
    → Linear(32)  → BN → ReLU
    → Linear(1)   → Sigmoid

The residual (skip) connections let gradients flow through the deeper
network without vanishing. Each skip connection uses a learned 1×1
linear projection to match dimensions.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from config.Settings import MODEL_PARAMS


class ResidualBlock(nn.Module):
    """Single hidden layer with optional skip connection.

    If input and output dimensions differ, a learned linear projection
    maps the skip path to the correct size.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.3,
        use_bn: bool = True,
        use_skip: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Skip connection — project if dimensions differ
        self.use_skip = use_skip and (in_features >= out_features)
        if self.use_skip and in_features != out_features:
            self.skip_proj = nn.Linear(in_features, out_features, bias=False)
        elif self.use_skip:
            self.skip_proj = nn.Identity()
        else:
            self.skip_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.bn(out)
        if self.use_skip and self.skip_proj is not None:
            out = out + self.skip_proj(x)
        out = self.act(out)
        out = self.drop(out)
        return out


class StrikePctNetwork(nn.Module):
    """MLP with residual connections for predicting strike percentage (0-1)."""

    def __init__(
        self,
        n_features: int,
        hidden_layers: Optional[List[int]] = None,
        dropout_rate: Optional[float] = None,
        use_batch_norm: Optional[bool] = None,
        use_residual: Optional[bool] = None,
    ):
        super().__init__()

        hidden = hidden_layers or MODEL_PARAMS.hidden_layers
        drop = dropout_rate if dropout_rate is not None else MODEL_PARAMS.dropout_rate
        bn = use_batch_norm if use_batch_norm is not None else MODEL_PARAMS.use_batch_norm
        skip = use_residual if use_residual is not None else MODEL_PARAMS.use_residual

        # Input normalization
        self.input_bn = nn.BatchNorm1d(n_features) if bn else nn.Identity()

        # Hidden blocks
        blocks: list[nn.Module] = []
        prev_size = n_features
        for i, h in enumerate(hidden):
            # Dropout on all but the last hidden layer
            d = drop if i < len(hidden) - 1 else 0.0
            blocks.append(
                ResidualBlock(prev_size, h, dropout=d, use_bn=bn, use_skip=skip)
            )
            prev_size = h

        self.hidden = nn.Sequential(*blocks)

        # Output head: single neuron with sigmoid
        self.head = nn.Sequential(
            nn.Linear(prev_size, 1),
            nn.Sigmoid(),
        )

        # Kaiming initialization
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = self.hidden(x)
        return self.head(x).squeeze(-1)