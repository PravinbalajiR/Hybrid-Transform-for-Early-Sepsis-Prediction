"""
organ_encoders.py
-----------------
Physiology-Aware Temporal Encoders (PATE).

Extracts explicitly engineered physiological features (slopes, ratios, rolling stats) 
from the raw (value, mask, delta) sequences.
Passes these through a Conv1D -> GRU -> Attention Pooling architecture to extract 
a single, rich representation token per organ system.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from preprocessing.load_data import ALL_FEATURE_COLS

def get_idx(name: str) -> int:
    return ALL_FEATURE_COLS.index(name)

def compute_slope(x: torch.Tensor) -> torch.Tensor:
    """Computes simple backward difference (x_t - x_{t-1}) over time dimension."""
    # x shape: (B, T)
    slope = torch.zeros_like(x)
    slope[:, 1:] = x[:, 1:] - x[:, :-1]
    return slope

def compute_rolling_mean(x: torch.Tensor, window: int = 3) -> torch.Tensor:
    """Computes a rolling mean using 1d average pooling with causal padding."""
    # x shape: (B, T)
    B, T = x.shape
    x_pad = F.pad(x.unsqueeze(1), (window-1, 0), mode='replicate') # (B, 1, T+window-1)
    return F.avg_pool1d(x_pad, kernel_size=window, stride=1).squeeze(1) # (B, T)

def compute_rolling_variance(x: torch.Tensor, window: int = 3) -> torch.Tensor:
    """Computes rolling variance."""
    mean_x = compute_rolling_mean(x, window)
    mean_x2 = compute_rolling_mean(x**2, window)
    var = mean_x2 - mean_x**2
    return F.relu(var) # Ensure positive due to numerical precision

class OrganAttentionPool(nn.Module):
    """Attention pooling to compress (B, T, d_model) -> (B, d_model)"""
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        attn_weights = F.softmax(self.attention(x), dim=1) # (B, T, 1)
        pooled = torch.sum(x * attn_weights, dim=1) # (B, d_model)
        return pooled

class TemporalOrganEncoder(nn.Module):
    """
    Conv1D -> GRU -> Attention Pooling
    """
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(32, d_model // 2)
        
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim)
        )
        # padding=2 on kernel=3 gives L+2. We'll slice the first L to make it causal
        
        # Causal GRU (unidirectional to prevent future leakage)
        self.gru = nn.GRU(hidden_dim, d_model, batch_first=True, bidirectional=False)
        self.pool = OrganAttentionPool(d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        returns: (B, d_model)
        """
        B, T, _ = x.shape
        
        # Conv1d expects (B, C, L)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)
        # Causal slice
        x_conv = x_conv[:, :, :T] 
        x_conv = x_conv.transpose(1, 2) # (B, T, hidden)
        
        # GRU
        gru_out, _ = self.gru(x_conv) # (B, T, d_model)
        gru_out = self.dropout(gru_out)
        
        # Attention Pool
        pooled = self.pool(gru_out) # (B, d_model)
        return self.norm(pooled)


class PhysiologyAwareTemporalEncoders(nn.Module):
    """
    Extracts explicit physiological features for 6 organ systems,
    and produces 6 Organ Tokens.
    """
    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.F_base = len(ALL_FEATURE_COLS)
        self.d_model = d_model
        
        # Define the number of engineered features for each organ
        # We also concatenate the Mask and Delta for the base variables
        self.organ_dims = {
            "cardiovascular": 7, # ShockIndex, PulsePressure, MAP slope, MAP, HR, SBP, DBP
            "respiratory": 5,    # Resp/O2 ratio, O2 deficit, O2Sat, Resp, O2Sat trend
            "renal": 5,          # BUN/Cr, Cr slope, Cr moving avg, BUN, Creatinine
            "liver": 3,          # AST/Bili, AST, Bilirubin
            "metabolic": 4,      # Lactate slope, pH, Lactate, BaseExcess
            "temperature": 1,    # Temp
        }
        
        self.mask_delta_dims = {
            "cardiovascular": 8,
            "respiratory": 4,
            "renal": 4,
            "liver": 4,
            "metabolic": 6,
            "temperature": 2,
        }
        
        self.encoders = nn.ModuleDict()
        self.organ_names = list(self.organ_dims.keys())
        for name in self.organ_names:
            in_dim = self.organ_dims[name] + self.mask_delta_dims[name]
            self.encoders[name] = TemporalOrganEncoder(input_dim=in_dim, d_model=d_model, dropout=dropout)

    def _get_mask_delta(self, x: torch.Tensor, var_names: List[str]) -> torch.Tensor:
        """Helper to extract masks and deltas for specific variables."""
        features = []
        for name in var_names:
            idx = get_idx(name)
            features.append(x[:, :, idx + self.F_base])       # mask
            features.append(x[:, :, idx + 2 * self.F_base])   # delta
        return torch.stack(features, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 3 * F_base)
        Returns: (B, 6, d_model)
        """
        # Values
        hr = x[:, :, get_idx("HR")]
        sbp = x[:, :, get_idx("SBP")]
        dbp = x[:, :, get_idx("DBP")]
        map_ = x[:, :, get_idx("MAP")]
        
        o2sat = x[:, :, get_idx("O2Sat")]
        resp = x[:, :, get_idx("Resp")]
        
        cr = x[:, :, get_idx("Creatinine")]
        bun = x[:, :, get_idx("BUN")]
        
        ast = x[:, :, get_idx("AST")]
        bili = x[:, :, get_idx("Bilirubin_total")]
        
        lac = x[:, :, get_idx("Lactate")]
        ph = x[:, :, get_idx("pH")]
        be = x[:, :, get_idx("BaseExcess")]
        
        temp = x[:, :, get_idx("Temp")]
        
        # --- Cardio ---
        # Correct Shock Index calculation: HR / (SBP + 1e-5) clipped to [0, 5]
        shock_index = torch.clamp(hr / (torch.abs(sbp) + 1e-5), min=0.0, max=5.0)
        pulse_pressure = sbp - dbp
        map_slope = compute_slope(map_)
        # Base features
        cardio_base = torch.stack([shock_index, pulse_pressure, map_slope, map_, hr, sbp, dbp], dim=-1)
        cardio_md = self._get_mask_delta(x, ["HR", "SBP", "MAP", "DBP"])
        cardio_feat = torch.cat([cardio_base, cardio_md], dim=-1)
        
        # --- Respiratory ---
        resp_o2_ratio = resp - o2sat
        o2_deficit = 100.0 - o2sat
        o2_trend = compute_slope(o2sat)
        resp_base = torch.stack([resp_o2_ratio, o2_deficit, o2_trend, o2sat, resp], dim=-1)
        resp_md = self._get_mask_delta(x, ["O2Sat", "Resp"])
        resp_feat = torch.cat([resp_base, resp_md], dim=-1)
        
        # --- Renal ---
        bun_cr = bun - cr
        cr_slope = compute_slope(cr)
        cr_avg = compute_rolling_mean(cr)
        renal_base = torch.stack([bun_cr, cr_slope, cr_avg, bun, cr], dim=-1)
        renal_md = self._get_mask_delta(x, ["Creatinine", "BUN"])
        renal_feat = torch.cat([renal_base, renal_md], dim=-1)
        
        # --- Liver ---
        ast_bili = ast - bili
        ast_trend = compute_slope(ast)
        bili_trend = compute_slope(bili)
        liver_base = torch.stack([ast_bili, ast_trend, bili_trend], dim=-1)
        liver_md = self._get_mask_delta(x, ["AST", "Bilirubin_total"])
        liver_feat = torch.cat([liver_base, liver_md], dim=-1)
        
        # --- Metabolic ---
        lac_slope = compute_slope(lac)
        ph_dev = torch.abs(7.4 - ph)
        metab_base = torch.stack([lac_slope, ph_dev, lac, be], dim=-1)
        metab_md = self._get_mask_delta(x, ["Lactate", "pH", "BaseExcess"])
        metab_feat = torch.cat([metab_base, metab_md], dim=-1)
        
        # --- Temperature ---
        temp_base = temp.unsqueeze(-1)
        temp_md = self._get_mask_delta(x, ["Temp"])
        temp_feat = torch.cat([temp_base, temp_md], dim=-1)
        
        # Process through Temporal Encoders
        tokens = []
        tokens.append(self.encoders["cardiovascular"](cardio_feat))
        tokens.append(self.encoders["respiratory"](resp_feat))
        tokens.append(self.encoders["renal"](renal_feat))
        tokens.append(self.encoders["liver"](liver_feat))
        tokens.append(self.encoders["metabolic"](metab_feat))
        tokens.append(self.encoders["temperature"](temp_feat))
        
        # Stack into (B, 6, d_model)
        out = torch.stack(tokens, dim=1)
        return out
