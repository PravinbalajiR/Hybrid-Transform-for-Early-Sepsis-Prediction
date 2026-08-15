"""
models/m5/value_encoder.py
--------------------------
Independent Neural Projection for 34 Physiological Value Features.
"""

import torch
import torch.nn as nn


class ValueEncoder(nn.Module):
    """Encodes the 34 continuous physiological measurement values."""
    
    def __init__(self, input_dim: int = 34, embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x_val: torch.Tensor) -> torch.Tensor:
        # x_val: [batch_size, seq_len, 34]
        return self.encoder(x_val)
