"""
models/m5/mask_encoder.py
-------------------------
Independent Neural Encoder for 34 Observation Missingness Masks.
"""

import torch
import torch.nn as nn


class MaskEncoder(nn.Module):
    """Encodes the 34 binary observation missingness masks."""
    
    def __init__(self, input_dim: int = 34, embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x_mask: torch.Tensor) -> torch.Tensor:
        # x_mask: [batch_size, seq_len, 34]
        return self.encoder(x_mask)
