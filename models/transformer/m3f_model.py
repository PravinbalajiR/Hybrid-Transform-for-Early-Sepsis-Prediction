"""
m3f_model.py
------------
M3-F: Final Time-Aware Transformer for Early Sepsis Prediction.

Novel Architecture:
  1. Value Embedding: Linear(34 -> 64) + LayerNorm + Dropout
  2. Mask Embedding: Linear(34 -> 64) + LayerNorm + Dropout
  3. Non-Periodic Log Time Encoder: log1p(delta) -> Linear(34 -> 64) -> GELU -> Linear(64 -> 64) -> LayerNorm
  4. Adaptive Softmax Fusion: Softmax(Linear([E_v, E_m, E_t])) -> [alpha, beta, gamma] (alpha + beta + gamma = 1.0)
     E_fused = alpha * E_v + beta * E_m + gamma * E_t
  5. LayerNorm + Sinusoidal Positional Encoding
  6. Transformer Encoder: 3 Layers, 4 Heads, d_model=64, dim_feedforward=128
  7. GELU Prediction Head: Linear(64 -> 32) -> GELU -> Dropout -> Linear(32 -> 1)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 64, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ValueEmbedding(nn.Module):
    def __init__(self, num_features: int = 34, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        out = self.proj(values)
        out = self.layer_norm(out)
        return self.dropout(out)


class MaskEmbedding(nn.Module):
    def __init__(self, num_features: int = 34, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        out = self.proj(masks)
        out = self.layer_norm(out)
        return self.dropout(out)


class LogTimeEncoder(nn.Module):
    def __init__(self, num_features: int = 34, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        # Non-periodic log transformation: log1p(t)
        log_deltas = torch.log1p(torch.clamp(deltas, min=0.0))
        return self.mlp(log_deltas)


class AdaptiveFusion(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        # Attention projection over concatenated [E_v, E_m, E_t] -> 3 weights
        self.weight_gate = nn.Sequential(
            nn.Linear(d_model * 3, 32),
            nn.GELU(),
            nn.Linear(32, 3)
        )

    def forward(
        self, E_v: torch.Tensor, E_m: torch.Tensor, E_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Shape of concatenated: (B, T, d_model * 3)
        cat_emb = torch.cat([E_v, E_m, E_t], dim=-1)
        
        # Calculate raw gating logits (B, T, 3)
        gate_logits = self.weight_gate(cat_emb)
        
        # Softmax over the 3 channels -> (alpha, beta, gamma), sum = 1.0
        weights = F.softmax(gate_logits, dim=-1)  # (B, T, 3)
        
        alpha = weights[..., 0:1]  # (B, T, 1)
        beta  = weights[..., 1:2]  # (B, T, 1)
        gamma = weights[..., 2:3]  # (B, T, 1)

        # Adaptive Weighted Combination
        E_fused = alpha * E_v + beta * E_m + gamma * E_t
        return E_fused, weights


class M3FinalModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 102,
        num_features: int = 34,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.value_embedding = ValueEmbedding(num_features=num_features, d_model=d_model, dropout=dropout)
        self.mask_embedding  = MaskEmbedding(num_features=num_features, d_model=d_model, dropout=dropout)
        self.time_encoder    = LogTimeEncoder(num_features=num_features, d_model=d_model, dropout=dropout)
        
        self.adaptive_fusion = AdaptiveFusion(d_model=d_model)
        self.fusion_norm     = nn.LayerNorm(d_model)
        self.pos_encoder     = PositionalEncoding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Split triplet input (B, T, 102) into Values (34), Masks (34), Deltas (34)
        v = x[..., 0 : self.num_features]
        m = x[..., self.num_features : 2 * self.num_features]
        d = x[..., 2 * self.num_features :]

        E_v = self.value_embedding(v)
        E_m = self.mask_embedding(m)
        E_t = self.time_encoder(d)

        # Adaptive Fusion
        E_fused, weights = self.adaptive_fusion(E_v, E_m, E_t)
        E_norm = self.pos_encoder(self.fusion_norm(E_fused))

        # Transformer Sequence Processing
        out = self.transformer_encoder(E_norm, src_key_padding_mask=padding_mask)
        logits = self.classifier(out).squeeze(-1)
        return logits
