"""
models/novelty/physio_transformer.py
------------------------------------
Physiology-Informed Time-Aware Causal Transformer (PITACT)

Core Proposed Architecture combining 4 Tightly Integrated Innovations:
  1. Strict Causal Temporal Self-Attention (zero future leakage)
  2. Informative Missingness & Temporal Reliability Decay: R_j(t) = exp(-gamma_j * delta_t_j)
  3. Causal Patient-Adaptive Baseline & Physiological Deterioration Dynamics (level, velocity, acceleration)
  4. Multi-Horizon Early-Warning Prediction Heads (6h, 12h, 24h)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class TemporalReliabilityEmbedding(nn.Module):
    """
    Computes mathematical observation temporal reliability:
      R_j(t) = exp(-gamma_j * delta_t_j(t))
    where gamma_j > 0 is a learnable per-variable decay parameter.
    """
    def __init__(self, num_features: int = 34):
        super().__init__()
        # Initialize gamma > 0
        self.gamma_log = nn.Parameter(torch.zeros(num_features))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t shape: (B, T, F)
        gamma = torch.exp(self.gamma_log).unsqueeze(0).unsqueeze(0) # (1, 1, F)
        reliability = torch.exp(-gamma * F.relu(delta_t))            # (B, T, F)
        return reliability


class CausalPhysiologicalDynamics(nn.Module):
    """
    Calculates causal physiological derivatives (velocity and acceleration) for irregularly sampled time-series.
    Uses strictly observations at or before time t (t' <= t).
    """
    def __init__(self, num_features: int = 34):
        super().__init__()
        self.num_features = num_features

    def forward(self, v: torch.Tensor, delta_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # v: (B, T, F), delta_t: (B, T, F)
        B, T, F = v.shape
        device = v.device

        # Causal backward difference for velocity: dv / dt
        dv = torch.zeros_like(v)
        dv[:, 1:, :] = v[:, 1:, :] - v[:, :-1, :]
        dt_safe = torch.clamp(delta_t, min=1.0)
        velocity = dv / dt_safe

        # Causal acceleration: d(velocity) / dt
        d_vel = torch.zeros_like(velocity)
        d_vel[:, 1:, :] = velocity[:, 1:, :] - velocity[:, :-1, :]
        acceleration = d_vel / dt_safe

        return velocity, acceleration


class CausalPatientBaseline(nn.Module):
    """
    Calculates causal running baseline mean for each patient:
      mu_patient(t) = (1 / t) * sum_{i=0}^t v(i)
    Ensures zero future information leakage.
    """
    def __init__(self, num_features: int = 34):
        super().__init__()

    def forward(self, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # v: (B, T, F), mask: (B, T, F)
        B, T, F = v.shape

        # Cumulative sum of observed values up to time t
        obs_v = v * mask
        cumsum_v = torch.cumsum(obs_v, dim=1)
        cumsum_mask = torch.cumsum(mask, dim=1).clamp(min=1.0)

        running_baseline = cumsum_v / cumsum_mask
        deviation = v - running_baseline
        return deviation


class DynamicOrganInteraction(nn.Module):
    """
    Learns dynamic organ-state interactions A_ij(t) between 6 organ nodes:
    Cardiovascular, Respiratory, Renal, Liver, Metabolic, Temperature.
    """
    def __init__(self, d_model: int = 64, num_organs: int = 6):
        super().__init__()
        self.num_organs = num_organs
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, organ_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # organ_tokens: (B, T, num_organs, d_model)
        B, T, O, D = organ_tokens.shape
        q = self.q_proj(organ_tokens) # (B, T, O, D)
        k = self.k_proj(organ_tokens) # (B, T, O, D)
        v = self.v_proj(organ_tokens) # (B, T, O, D)

        # Dynamic interaction matrix per timestep
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5) # (B, T, O, O)
        A = F.softmax(scores, dim=-1)

        fused_organs = torch.matmul(A, v) # (B, T, O, D)
        fused_representation = fused_organs.mean(dim=2) # (B, T, D)
        return fused_representation, A


class PITACTModel(nn.Module):
    """
    Physiology-Informed Time-Aware Causal Transformer (PITACT)
    """
    def __init__(
        self,
        num_raw_features: int = 34,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        enable_reliability: bool = True,
        enable_dynamics: bool = True,
        enable_multihorizon: bool = True,
    ):
        super().__init__()
        self.num_raw_features = num_raw_features
        self.d_model = d_model
        self.enable_reliability = enable_reliability
        self.enable_dynamics = enable_dynamics
        self.enable_multihorizon = enable_multihorizon

        # Core components
        self.reliability_layer = TemporalReliabilityEmbedding(num_features=num_raw_features)
        self.dynamics_layer = CausalPhysiologicalDynamics(num_features=num_raw_features)
        self.baseline_layer = CausalPatientBaseline(num_features=num_raw_features)

        # Input dimension calculator:
        # Base: values [34], masks [34], deltas [34] = 102
        # Optional: reliability [34], velocity [34], acceleration [34], baseline_dev [34]
        in_dim = 34 * 3
        if enable_reliability:
            in_dim += 34
        if enable_dynamics:
            in_dim += 34 * 3 # velocity, acceleration, baseline deviation

        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))

        # Causal Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-Horizon Heads (6h primary, 12h, 24h secondary)
        self.head_6h = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        if enable_multihorizon:
            self.head_12h = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            )
            self.head_24h = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            )

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, 102) -> [vals, masks, deltas]
        padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        v = x[:, :, 0:34]
        m = x[:, :, 34:68]
        dt = x[:, :, 68:102]

        feature_list = [v, m, dt]

        if self.enable_reliability:
            r = self.reliability_layer(dt)
            feature_list.append(r)

        if self.enable_dynamics:
            vel, acc = self.dynamics_layer(v, dt)
            b_dev = self.baseline_layer(v, m)
            feature_list.extend([vel, acc, b_dev])

        # Concatenate engineered features
        x_concat = torch.cat(feature_list, dim=-1) # (B, T, in_dim)

        # Projection + Positional encoding
        h = self.input_proj(x_concat) + self.pos_encoder[:, :T, :]

        # Strict Causal Transformer Encoding
        causal_mask = self._generate_causal_mask(T, x.device)
        h_out = self.transformer_encoder(h, mask=causal_mask, src_key_padding_mask=padding_mask)

        # Compute predictions
        logits_6h = self.head_6h(h_out).squeeze(-1)
        out = {"logits": logits_6h, "logits_6h": logits_6h}

        if self.enable_multihorizon:
            out["logits_12h"] = self.head_12h(h_out).squeeze(-1)
            out["logits_24h"] = self.head_24h(h_out).squeeze(-1)

        return out
