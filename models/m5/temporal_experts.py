"""
models/m5/temporal_experts.py
------------------------------
Specialized Temporal Experts for M5:
  1. LocalTemporalExpert: Causal 1D TCN for short-term physiological shifts.
  2. GlobalTemporalExpert: Causal Transformer Encoder for long-range context.
  3. TimeAwareExpert: Irregular observation timing MLP expert.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D Causal Convolution ensuring zero lookahead bias."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_channels, seq_len]
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]  # Slice off future padding
        return out


class ChannelLayerNorm(nn.Module):
    """LayerNorm across channels for 1D convolutions without cross-timestep leakage."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, seq_len] -> transpose -> [batch, seq_len, channels]
        x_t = x.transpose(1, 2)
        out = self.ln(x_t)
        return out.transpose(1, 2)


class LocalTemporalExpert(nn.Module):
    """Causal TCN for capturing short-term local physiological shifts (e.g. rapid vital spikes)."""
    
    def __init__(self, in_dim: int = 64, hidden_dim: int = 64, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_dim, hidden_dim, kernel_size=kernel_size, dilation=1)
        self.ln1 = ChannelLayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=2)
        self.ln2 = ChannelLayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, in_dim]
        # Transpose for Conv1d: [batch, in_dim, seq_len]
        h = x.transpose(1, 2)
        
        res = h
        h = self.drop1(self.act1(self.ln1(self.conv1(h))))
        h = self.drop2(self.act2(self.ln2(self.conv2(h))))
        h = h + res  # Residual connection
        
        # Transpose back: [batch, seq_len, hidden_dim]
        out = h.transpose(1, 2)
        return self.proj(out)


class GlobalTemporalExpert(nn.Module):
    """Causal Transformer Encoder for capturing long-range multi-variable trajectory context."""
    
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return mask
        
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        causal_mask = self._generate_causal_mask(seq_len, x.device)
        return self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)


class TimeAwareExpert(nn.Module):
    """Specialized Expert emphasizing irregular temporal sampling patterns and deltas."""
    
    def __init__(self, time_dim: int = 32, val_mask_dim: int = 64, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim + val_mask_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def forward(self, h_time: torch.Tensor, h_val_mask: torch.Tensor) -> torch.Tensor:
        # h_time: [batch, seq_len, 32], h_val_mask: [batch, seq_len, 64]
        cat = torch.cat([h_time, h_val_mask], dim=-1)
        return self.mlp(cat)
