"""
models/m5/time_encoder.py
-------------------------
Irregular Elapsed Time Delta Encoder using log1p transformation.

Documentation:
- Units of delta: Elapsed ICU hours since last variable observation.
- Transformation: log1p(clamp(delta, min=0.0, max=168.0))
- Normalization: LayerNorm(embed_dim)
- Clipping: Max 168.0 hours
- Initial / Missing handling: 0.0 hours elapsed (first timestep).
"""

import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Encodes irregular elapsed time deltas."""
    
    def __init__(self, input_dim: int = 34, embed_dim: int = 32, dropout: float = 0.1, max_clip: float = 168.0):
        super().__init__()
        self.max_clip = max_clip
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x_delta: torch.Tensor) -> torch.Tensor:
        # x_delta: [batch_size, seq_len, 34]
        # Clamped log1p transformation
        x_clamped = torch.clamp(x_delta, min=0.0, max=self.max_clip)
        x_log = torch.log1p(x_clamped)
        return self.encoder(x_log)
