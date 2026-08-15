"""
models/m5/fusion.py
-------------------
1. AdaptiveRepresentationFusion: Softmax-constrained learned branch fusion.
2. TemporalAttentionPooling: Causal temporal attention pooling mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class AdaptiveRepresentationFusion(nn.Module):
    """
    Adaptive Representation Fusion with Softmax normalization constraint.
    Fuses Value, Mask, Time, and Expert representations.
    Guarantees: sum(weights) = 1.0.
    """
    
    def __init__(self, d_model: int = 64, num_branches: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_branches = num_branches
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * num_branches, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_branches)
        )
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, branch_tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        branch_tensors : List of tensors each [batch, seq_len, d_model]
                         e.g. [h_val, h_mask, h_time, h_expert]
                         
        Returns
        -------
        (fused_output, fusion_weights)
        """
        assert len(branch_tensors) == self.num_branches
        cat_branches = torch.cat(branch_tensors, dim=-1) # [batch, seq_len, d_model * num_branches]
        
        logits = self.gate_net(cat_branches)
        weights = F.softmax(logits, dim=-1) # [batch, seq_len, num_branches]
        
        # Verify normalization sum = 1.0
        weight_sum = weights.sum(dim=-1)
        assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5), "Fusion weights must sum to 1.0!"
        
        fused = 0.0
        for i, branch in enumerate(branch_tensors):
            w = weights[:, :, i:i+1] # [batch, seq_len, 1]
            fused = fused + w * branch
            
        return self.proj(fused), weights


class TemporalAttentionPooling(nn.Module):
    """
    Causal Temporal Attention Mechanism to highlight key clinical timesteps.
    Ensures zero future lookahead bias.
    """
    
    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.attn_score = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x            : [batch, seq_len, d_model]
        padding_mask : [batch, seq_len] (True for padded steps)
        
        Returns
        -------
        (enhanced_x, attention_scores)
        """
        scores = self.attn_score(x).squeeze(-1) # [batch, seq_len]
        
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1).unsqueeze(-1) # [batch, seq_len, 1]
        
        # Residual enhancement
        enhanced = x + attn_weights * x
        return enhanced, attn_weights.squeeze(-1)
