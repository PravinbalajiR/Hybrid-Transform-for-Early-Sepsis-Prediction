"""
time_aware_embedding.py
-----------------------
Time-Aware and Missingness-Aware Embedding Layer for ICU Time-Series.

Transforms raw hourly input triplets (values, observation mask, time delta)
into a dense embedding vector in d_model space:

  1. Feature Projection: Projects 3*F triplet vector [x_imputed, mask, time_delta] -> d_model
  2. Sinusoidal Positional Encoding: Encodes temporal sequence position t
  3. Continuous Time-Delta Embedding: Projects variable-specific time gaps into d_model

Reference:
  Vaswani et al. (2017) "Attention Is All You Need"
  Che et al. (2018) "Recurrent Neural Networks for Multivariate Time Series with Missing Values"
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard Sinusoidal Positional Encoding for time-series sequences."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Time2Vec(nn.Module):
    """
    Time2Vec: Learning a Vector Representation of Time
    Transforms continuous time features (deltas) into a robust frequency representation.
    """
    def __init__(self, in_features: int, k: int = 4):
        super().__init__()
        self.k = k
        self.in_features = in_features
        
        # Linear component
        self.w0 = nn.Parameter(torch.randn(1, 1, in_features))
        self.b0 = nn.Parameter(torch.randn(1, 1, in_features))
        
        # Periodic component (sine waves)
        self.w_p = nn.Parameter(torch.randn(1, 1, in_features, k - 1))
        self.b_p = nn.Parameter(torch.randn(1, 1, in_features, k - 1))
        
    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """tau shape: (B, T, F)"""
        v0 = tau * self.w0 + self.b0
        v0 = v0.unsqueeze(-1) # (B, T, F, 1)
        
        tau_p = tau.unsqueeze(-1) # (B, T, F, 1)
        v_p = torch.sin(tau_p * self.w_p + self.b_p) # (B, T, F, k-1)
        
        out = torch.cat([v0, v_p], dim=-1) # (B, T, F, k)
        B, T, F, K = out.shape
        return out.view(B, T, F * K)


class TimeAwareEmbedding(nn.Module):
    """
    Time-Aware Embedding module.
    Projects triplet (values, mask, delta) -> d_model using Time2Vec for deltas.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        max_len: int = 500,
        dropout: float = 0.1,
        ablation_mode: str = "none",  # "none", "linear_delta"
    ):
        super().__init__()
        self.d_model = d_model
        self.ablation_mode = ablation_mode
        self.is_triplet = (input_dim % 3 == 0) and (input_dim >= 102)
        self.F = input_dim // 3 if self.is_triplet else input_dim

        if self.is_triplet and ablation_mode != "linear_delta":
            self.use_time2vec = True
            self.t2v_k = 4
            self.time2vec = Time2Vec(in_features=self.F, k=self.t2v_k)
            # 2*F for values and mask + F*k for Time2Vec delta
            proj_in_dim = 2 * self.F + self.F * self.t2v_k
            self.proj = nn.Linear(proj_in_dim, d_model)
        else:
            self.use_time2vec = False
            self.proj = nn.Linear(input_dim, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_time2vec:
            values = x[:, :, :self.F]
            masks  = x[:, :, self.F:2*self.F]
            deltas = x[:, :, 2*self.F:]

            # Apply ablation masking logic
            if self.ablation_mode == "mask_only":
                # Zero out time deltas (only values + masks active)
                deltas = torch.zeros_like(deltas)
            elif self.ablation_mode == "delta_only":
                # Zero out observation masks (only values + deltas active)
                masks = torch.zeros_like(masks)

            # Apply log1p temporal scaling to compress continuous ICU time intervals [0, 50+h]
            scaled_deltas = torch.log1p(deltas)

            delta_emb = self.time2vec(scaled_deltas)
            x_proj_in = torch.cat([values, masks, delta_emb], dim=-1)
            out = self.proj(x_proj_in) * math.sqrt(self.d_model)

        else:
            out = self.proj(x) * math.sqrt(self.d_model)

        out = self.layer_norm(out)
        out = self.pos_encoder(out)
        return out

