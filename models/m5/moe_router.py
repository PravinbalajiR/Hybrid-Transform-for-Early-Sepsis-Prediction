"""
models/m5/moe_router.py
-----------------------
Adaptive Mixture-of-Experts Router with Softmax-constrained weights.
Provides interpretability tracking for expert utilization and routing entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class MoERouter(nn.Module):
    """
    Lightweight Adaptive Expert Router.
    Computes normalized expert weights: w = Softmax(G(h))
    where w_1 + w_2 + w_3 = 1.
    """
    
    def __init__(self, in_dim: int = 64, num_experts: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.gating = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_experts)
        )
        
    def forward(
        self,
        h_shared: torch.Tensor,
        e_local: torch.Tensor,
        e_global: torch.Tensor,
        e_time: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        h_shared  : Shared patient-hour representation [batch, seq_len, in_dim]
        e_local   : Local temporal expert output [batch, seq_len, expert_dim]
        e_global  : Global temporal expert output [batch, seq_len, expert_dim]
        e_time    : Time-aware expert output [batch, seq_len, expert_dim]
        
        Returns
        -------
        (fused_experts, routing_info_dict)
        """
        # Compute gating logits: [batch, seq_len, num_experts]
        logits = self.gating(h_shared)
        weights = F.softmax(logits, dim=-1)  # Constrained sum to 1.0
        
        # Expert weighting
        w_local  = weights[:, :, 0:1]
        w_global = weights[:, :, 1:2]
        w_time   = weights[:, :, 2:3]
        
        fused = w_local * e_local + w_global * e_global + w_time * e_time
        
        # Compute routing entropy for interpretability: -sum(w * log(w + 1e-8))
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        
        routing_info = {
            "weights": weights,               # [batch, seq_len, 3]
            "w_local": w_local.squeeze(-1),   # [batch, seq_len]
            "w_global": w_global.squeeze(-1), # [batch, seq_len]
            "w_time": w_time.squeeze(-1),     # [batch, seq_len]
            "entropy": entropy                # [batch, seq_len]
        }
        
        return fused, routing_info
