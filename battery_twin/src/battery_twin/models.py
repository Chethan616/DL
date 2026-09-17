"""Comparable sequence models for the NASA-first experiment."""

from __future__ import annotations

import torch
from torch import nn


class IntraCycleEncoder(nn.Module):
    """Encode voltage/current/temperature samples inside one cycle.

    Masked pooling is intentionally simple and auditable for the first POC.
    The real `delta_t` feature carries irregular timing without interpolation.
    """

    def __init__(self, input_features: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        # x: [batch, history, points, features]
        batch, history, points, _ = x.shape
        encoded = self.projection(x.reshape(batch * history, points, -1))
        mask = point_mask.reshape(batch * history, points).bool().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1)
        mean = (encoded * mask).sum(dim=1) / denom
        max_values = encoded.masked_fill(~mask, -1e4).max(dim=1).values
        pooled = 0.5 * (mean + max_values)
        return pooled.reshape(batch, history, -1)


class _HealthHeads(nn.Module):
    def __init__(self, hidden_size: int, rul_scale: float):
        super().__init__()
        self.soh_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.GELU(), nn.Linear(hidden_size // 2, 1))
        self.rul_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.GELU(), nn.Linear(hidden_size // 2, 1))
        self.rul_scale = float(rul_scale)

    def predict(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        # Bounded health and non-negative RUL are part of the model contract.
        soh = 100.0 * torch.sigmoid(self.soh_head(sequence).squeeze(-1))
        # Bound RUL to the configured horizon so early training cannot emit
        # physically impossible multi-thousand-cycle estimates.
        rul = self.rul_scale * torch.sigmoid(self.rul_head(sequence).squeeze(-1))
        return {"soh": soh, "rul": rul, "embedding": sequence}


class AdaptiveBatteryTwin(nn.Module):
    """Intra-cycle encoder + Transformer history encoder + SOH/RUL heads."""

    def __init__(
        self,
        input_features: int = 5,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        rul_scale: float = 250.0,
        max_history: int = 64,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.cycle_encoder = IntraCycleEncoder(input_features, hidden_size, dropout)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_history, hidden_size))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_heads = _HealthHeads(hidden_size, rul_scale)

    def forward(self, x: torch.Tensor, point_mask: torch.Tensor, cycle_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        cycle_embedding = self.cycle_encoder(x, point_mask)
        if cycle_embedding.shape[1] > self.position_embedding.shape[1]:
            raise ValueError("history length exceeds the model positional-embedding capacity")
        cycle_embedding = cycle_embedding + self.position_embedding[:, :cycle_embedding.shape[1]]
        padding_mask = ~cycle_mask.bool()
        sequence = self.temporal_encoder(cycle_embedding, src_key_padding_mask=padding_mask)
        return self.output_heads.predict(sequence)


class FeedForwardTwin(nn.Module):
    """Per-cycle DNN baseline; no temporal attention or recurrence."""

    def __init__(self, input_features: int = 5, hidden_size: int = 128, dropout: float = 0.1, rul_scale: float = 250.0):
        super().__init__()
        self.cycle_encoder = IntraCycleEncoder(input_features, hidden_size, dropout)
        self.output_heads = _HealthHeads(hidden_size, rul_scale)

    def forward(self, x: torch.Tensor, point_mask: torch.Tensor, cycle_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.output_heads.predict(self.cycle_encoder(x, point_mask))


class GRUBatteryTwin(nn.Module):
    """Recurrent sequence baseline with the same inputs and heads."""

    def __init__(self, input_features: int = 5, hidden_size: int = 128, dropout: float = 0.1, rul_scale: float = 250.0):
        super().__init__()
        self.cycle_encoder = IntraCycleEncoder(input_features, hidden_size, dropout)
        self.temporal_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_heads = _HealthHeads(hidden_size, rul_scale)

    def forward(self, x: torch.Tensor, point_mask: torch.Tensor, cycle_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        sequence = self.temporal_encoder(self.cycle_encoder(x, point_mask))[0]
        return self.output_heads.predict(sequence)


def build_model(name: str, **kwargs) -> nn.Module:
    """Factory used by the comparison runner."""

    key = name.lower().replace("_", "-")
    if key in {"dnn", "feedforward", "feed-forward"}:
        return FeedForwardTwin(
            input_features=kwargs["input_features"],
            hidden_size=kwargs["hidden_size"],
            dropout=kwargs.get("dropout", 0.1),
            rul_scale=kwargs.get("rul_scale", 250.0),
        )
    if key in {"gru", "lstm"}:
        # GRU is the first reproducible recurrent baseline; the interface lets
        # the experiment label it explicitly as GRU.
        return GRUBatteryTwin(
            input_features=kwargs["input_features"],
            hidden_size=kwargs["hidden_size"],
            dropout=kwargs.get("dropout", 0.1),
            rul_scale=kwargs.get("rul_scale", 250.0),
        )
    if key in {"transformer", "transformer-only", "pinn", "adaptive", "proposed"}:
        return AdaptiveBatteryTwin(**kwargs)
    raise ValueError(f"Unknown model '{name}'. Choose dnn, gru, transformer, pinn, or adaptive.")
