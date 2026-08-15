"""
m5_model.py
-----------
M5: Multi-Hybrid Time-Aware Sepsis Intelligence Network.
Unified Modular Architecture supporting Staged Variants (M5-A, M5-B, M5-C, M5-D, M5-FINAL)
and Component Ablation Studies.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

from models.m5.value_encoder import ValueEncoder
from models.m5.mask_encoder import MaskEncoder
from models.m5.time_encoder import TimeEncoder
from models.m5.temporal_experts import LocalTemporalExpert, GlobalTemporalExpert, TimeAwareExpert
from models.m5.moe_router import MoERouter
from models.m5.fusion import AdaptiveRepresentationFusion, TemporalAttentionPooling


class M5Model(nn.Module):
    """
    M5 Architecture: Multi-Hybrid Time-Aware Sepsis Intelligence Network.
    
    Inputs:
      - x: Triplet vector [batch_size, seq_len, 102]
           Contains: Values [34], Masks [34], Time Deltas [34].
    """
    
    def __init__(
        self,
        input_dim: int = 102,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        kernel_size: int = 3,
        variant: str = "M5-FINAL"
    ):
        super().__init__()
        self.variant = variant
        self.d_model = d_model
        
        # Encoders
        self.val_encoder  = ValueEncoder(input_dim=34, embed_dim=32, dropout=dropout)
        self.mask_encoder = MaskEncoder(input_dim=34, embed_dim=32, dropout=dropout)
        self.time_encoder = TimeEncoder(input_dim=34, embed_dim=32, dropout=dropout)
        
        # Branch projections to d_model (64)
        self.val_proj  = nn.Linear(32, d_model)
        self.mask_proj = nn.Linear(32, d_model)
        self.time_proj = nn.Linear(32, d_model)
        
        # Shared input projection
        self.shared_proj = nn.Linear(32 * 3, d_model)
        
        # Experts
        self.local_expert  = LocalTemporalExpert(in_dim=d_model, hidden_dim=d_model, kernel_size=kernel_size, dropout=dropout)
        self.global_expert = GlobalTemporalExpert(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.time_expert   = TimeAwareExpert(time_dim=32, val_mask_dim=d_model, hidden_dim=d_model, dropout=dropout)
        
        # Router
        self.moe_router = MoERouter(in_dim=d_model, num_experts=3, dropout=dropout)
        
        # Adaptive Fusion & Attention Pooling
        self.adaptive_fusion = AdaptiveRepresentationFusion(d_model=d_model, num_branches=4, dropout=dropout)
        self.attn_pooling    = TemporalAttentionPooling(d_model=d_model, dropout=dropout)
        
        # Compact Classifier Prediction Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        x            : Input tensor [batch_size, seq_len, 102]
        padding_mask : Optional padding mask [batch_size, seq_len]
        
        Returns
        -------
        (logits, interpretability_info_dict)
        """
        # 1. Unpack input features
        x_val   = x[:, :, 0:34]
        x_mask  = x[:, :, 34:68]
        x_delta = x[:, :, 68:102]
        
        # 2. Independent Branch Encoding
        e_val  = self.val_encoder(x_val)    # [batch, seq_len, 32]
        
        if self.variant == "no_mask":
            e_mask = torch.zeros_like(e_val)
        else:
            e_mask = self.mask_encoder(x_mask)  # [batch, seq_len, 32]
            
        if self.variant == "no_time":
            e_time = torch.zeros_like(e_val)
        else:
            e_time = self.time_encoder(x_delta) # [batch, seq_len, 32]
            
        # Branch projections
        h_val  = self.val_proj(e_val)   # [batch, seq_len, 64]
        h_mask = self.mask_proj(e_mask) # [batch, seq_len, 64]
        h_time = self.time_proj(e_time) # [batch, seq_len, 64]
        
        # Shared input representation
        h_shared = self.shared_proj(torch.cat([e_val, e_mask, e_time], dim=-1)) # [batch, seq_len, 64]
        
        # 3. Temporal Experts Execution
        # Local TCN Expert
        if self.variant in ["M5-A", "no_cnn"]:
            e_local_out = h_shared
        else:
            e_local_out = self.local_expert(h_shared)
            
        # Global Transformer Expert
        if self.variant == "no_transformer":
            e_global_out = h_shared
        else:
            e_global_out = self.global_expert(h_shared, padding_mask=padding_mask)
            
        # Time-Aware Expert
        if self.variant in ["M5-A", "M5-B"]:
            e_time_out = h_shared
        else:
            h_val_mask = h_val + h_mask
            e_time_out = self.time_expert(e_time, h_val_mask)
            
        # 4. Expert Routing & Fusion
        routing_info = {}
        if self.variant in ["M5-A", "M5-B", "M5-C", "no_moe"]:
            # Equal weighting average fallback
            h_expert = (e_local_out + e_global_out + e_time_out) / 3.0
        else:
            h_expert, routing_info = self.moe_router(h_shared, e_local_out, e_global_out, e_time_out)
            
        # 5. Representation Fusion
        fusion_weights = None
        if self.variant in ["M5-A", "M5-B", "M5-C", "M5-D", "no_fusion"]:
            h_fused = (h_val + h_mask + h_time + h_expert) / 4.0
        else:
            h_fused, fusion_weights = self.adaptive_fusion([h_val, h_mask, h_time, h_expert])
            
        # 6. Temporal Attention Pooling
        attn_scores = None
        if self.variant in ["M5-A", "M5-B", "M5-C", "M5-D", "no_attention"]:
            h_final = h_fused
        else:
            h_final, attn_scores = self.attn_pooling(h_fused, padding_mask=padding_mask)
            
        # 7. Final Prediction Head
        logits = self.classifier(h_final).squeeze(-1) # [batch, seq_len]
        
        info = {
            "routing_info": routing_info,
            "fusion_weights": fusion_weights,
            "attn_scores": attn_scores
        }
        
        return logits, info
