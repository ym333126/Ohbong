"""Transformer seq2seq model used by the Ohbong prediction module.

The model receives one reservoir's recent time-series features and a reservoir
index, then directly predicts the next ``pred_len`` daily storage rates.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first time-series tensors."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional information to ``x`` with shape ``(batch, seq_len, d_model)``."""

        if x.dim() != 3:
            raise ValueError(f"Expected x to be 3D (batch, seq_len, d_model), got {tuple(x.shape)}")
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional encoding max_len {self.pe.size(1)}"
            )
        return self.dropout(x + self.pe[:, : x.size(1), :])


class Seq2SeqTransformer(nn.Module):
    """Encoder-based Transformer for 30-day reservoir storage-rate forecasting.

    Parameters are intentionally named to match common keys in
    ``transformer_v4_seq2seq_config.json``. Use :meth:`from_config` when loading
    the model from the prediction module so small config-key differences are
    handled in one place.
    """

    def __init__(
        self,
        input_dim: int = 16,
        num_reservoirs: int | None = None,
        n_reservoirs: int = 22,
        seq_len: int = 60,
        pred_len: int = 30,
        d_model: int = 128,
        nhead: int | None = None,
        n_heads: int = 4,
        num_encoder_layers: int | None = None,
        n_layers: int = 3,
        dim_feedforward: int | None = None,
        d_ff: int = 256,
        dropout: float = 0.3,
        reservoir_emb_dim: int | None = None,
        emb_dim: int = 8,
        max_len: int = 512,
        activation: str = "gelu",
        norm_first: bool = True,
        output_clip: bool = False,
    ) -> None:
        super().__init__()
        num_reservoirs = n_reservoirs if num_reservoirs is None else num_reservoirs
        nhead = n_heads if nhead is None else nhead
        num_encoder_layers = n_layers if num_encoder_layers is None else num_encoder_layers
        dim_feedforward = d_ff if dim_feedforward is None else dim_feedforward
        reservoir_emb_dim = emb_dim if reservoir_emb_dim is None else reservoir_emb_dim

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_reservoirs <= 0:
            raise ValueError("num_reservoirs must be positive")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")

        self.input_dim = input_dim
        self.num_reservoirs = num_reservoirs
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.output_clip = output_clip

        # Keep module names aligned with transformer_v4_seq2seq.pt state_dict.
        self.reservoir_emb = nn.Embedding(num_reservoirs, reservoir_emb_dim)
        self.input_proj = nn.Linear(input_dim + reservoir_emb_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max(seq_len, max_len), dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Seq2SeqTransformer":
        """Build a model from ``transformer_v4_seq2seq_config.json``.

        The helper accepts several key aliases so the inference code stays
        stable even if the notebook saved slightly different config names.
        """

        config = config.get("model_config", config.get("model", config))

        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in config:
                    return config[key]
            return default

        def as_bool(value: Any) -> bool:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y"}
            return bool(value)

        feature_columns = pick("feature_cols", "feature_columns", "numeric_features", "features", default=None)
        reservoir_mapping = pick("reservoir_to_id", "reservoir_encoder", "reservoir_mapping", default=None)

        input_dim = pick("input_dim", "num_features", "num_numeric_features")
        if input_dim is None and feature_columns is not None:
            input_dim = len(feature_columns)

        num_reservoirs = pick("num_reservoirs", "n_reservoirs", "reservoir_vocab_size")
        if num_reservoirs is None and reservoir_mapping is not None:
            num_reservoirs = len(reservoir_mapping)

        if input_dim is None:
            raise KeyError("Config must include input_dim/num_features or feature_columns.")
        if num_reservoirs is None:
            raise KeyError("Config must include num_reservoirs/reservoir_vocab_size or reservoir mapping.")

        # Make sure the embedding table covers the largest id in the saved mapping.
        max_reservoir_id = None
        if isinstance(reservoir_mapping, Mapping) and reservoir_mapping:
            try:
                max_reservoir_id = max(int(v) for v in reservoir_mapping.values())
            except (TypeError, ValueError):
                max_reservoir_id = None
        if max_reservoir_id is not None:
            num_reservoirs = max(int(num_reservoirs), max_reservoir_id + 1)

        return cls(
            input_dim=int(input_dim),
            num_reservoirs=int(num_reservoirs),
            seq_len=int(pick("seq_len", "input_seq_len", "sequence_length", default=60)),
            pred_len=int(pick("pred_len", "forecast_horizon", "horizon", "output_len", default=30)),
            d_model=int(pick("d_model", "hidden_dim", "model_dim", default=128)),
            nhead=int(pick("nhead", "n_head", "n_heads", "num_heads", default=4)),
            num_encoder_layers=int(pick("num_encoder_layers", "n_layers", "num_layers", "encoder_layers", default=3)),
            dim_feedforward=int(pick("dim_feedforward", "d_ff", "ff_dim", "feedforward_dim", default=256)),
            dropout=float(pick("dropout", "dropout_rate", default=0.1)),
            reservoir_emb_dim=int(pick("reservoir_emb_dim", "reservoir_embedding_dim", "emb_dim", default=16)),
            max_len=int(pick("max_len", "max_seq_len", "max_position_embeddings", default=512)),
            activation=str(pick("activation", default="gelu")),
            norm_first=as_bool(pick("norm_first", default=True)),
            output_clip=as_bool(pick("output_clip", "clip_output", default=False)),
        )

    def forward(self, x: torch.Tensor, reservoir_id: torch.Tensor) -> torch.Tensor:
        """Predict future storage rates.

        Args:
            x: Numeric model input with shape ``(batch, seq_len, input_dim)``.
            reservoir_id: Reservoir embedding index with shape ``(batch,)`` or
                ``(batch, 1)``.

        Returns:
            Tensor of predicted storage rates with shape ``(batch, pred_len)``.
        """

        if x.dim() != 3:
            raise ValueError(f"Expected x shape (batch, seq_len, input_dim), got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")

        batch_size, seq_len, _ = x.shape
        if reservoir_id.dim() == 2 and reservoir_id.size(1) > 1:
            reservoir_id = reservoir_id[:, -1]
        reservoir_id = reservoir_id.reshape(batch_size).long()

        reservoir_seq = self.reservoir_emb(reservoir_id).unsqueeze(1).expand(batch_size, seq_len, -1)
        x = torch.cat([x, reservoir_seq], dim=-1)
        x = self.input_proj(x)
        x = self.pos_enc(x)

        encoded = self.encoder(x)
        forecast = self.head(encoded[:, -1, :])
        if self.output_clip:
            forecast = forecast.clamp(0.0, 100.0)
        return forecast
