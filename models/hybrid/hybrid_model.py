"""
hybrid_model.py
---------------
The M4 v2 Knowledge-Guided Hybrid Transformer.

Architecture:
1. Organ Branch (PATE): Extracts 6 Organ Tokens using Conv1D->GRU->Attention Pooling
2. Temporal Branch: Pre-pends the 6 Organ Tokens to the time-series embedding.
3. Transformer: Learns shared representation across organs and time.
4. Multi-Task Heads: 
   - Sepsis Prediction Head
   - Physiological Forecasting Head (predicts next-hour deltas)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from models.transformer.tact_model import TACTModel as SepsisTransformer
from models.organ_branch.organ_encoders import PhysiologyAwareTemporalEncoders


class SepsisHybridModel(nn.Module):
    """
    M4 v2: Token-Injected, Self-Supervised Knowledge-Guided Hybrid Transformer.
    """
    def __init__(
        self,
        input_dim: int = 102,       # Triplet input (34 * 3)
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 500,
        ablation_mode: str = "none" # 'no_prefix', 'no_forecast'
    ):
        super().__init__()
        
        self.ablation_mode = ablation_mode
        self.d_model = d_model
        
        # 1. Temporal Branch (Initializes from M3 logic)
        self.temporal_branch = SepsisTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len + 6 # +6 for the organ tokens
        )
        
        # 2. Knowledge Branch (OATEs)
        self.organ_branch = PhysiologyAwareTemporalEncoders(d_model=d_model, dropout=dropout)
        
        # 3. Final Prediction Head (Sepsis)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        
        # 4. Forecasting Head (predicts 5 variables: MAP, Cr, Lactate, O2Sat, RespRate)
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
        # In case of late fusion ablation
        if ablation_mode == "no_prefix":
            self.late_fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, 102) triplet tensor
        padding_mask: (B, T) boolean mask
        """
        B, T, _ = x.shape
        
        # --- Knowledge Branch ---
        # organ_tokens: (B, 6, d_model)
        organ_tokens = self.organ_branch(x)
        
        # --- Temporal Branch Embedding ---
        h_temp = self.temporal_branch.embedding(x) # (B, T, d_model)
        
        if self.ablation_mode == "no_prefix":
            # Baseline: M4 without prefix tokens (late fusion)
            out = self.temporal_branch.transformer_encoder(h_temp, src_key_padding_mask=padding_mask)
            # Global average of organ tokens
            organ_global = organ_tokens.mean(dim=1).unsqueeze(1).expand(-1, T, -1) # (B, T, d_model)
            out = self.late_fusion(torch.cat([out, organ_global], dim=-1))
            
        else:
            # Proposed: Prefix Token Injection
            # Prepend organ tokens to the sequence
            # (B, T+6, d_model)
            seq = torch.cat([organ_tokens, h_temp], dim=1)
            
            # Extend padding mask (organs are never padded)
            if padding_mask is not None:
                organ_mask = torch.zeros((B, 6), dtype=torch.bool, device=x.device)
                extended_mask = torch.cat([organ_mask, padding_mask], dim=1) # (B, T+6)
            else:
                extended_mask = None
                
            # Pass through Transformer
            out_seq = self.temporal_branch.transformer_encoder(seq, src_key_padding_mask=extended_mask)
            
            # Slice back the temporal states
            out = out_seq[:, 6:, :] # (B, T, d_model)
        
        # --- Prediction Heads ---
        sepsis_logits = self.fc_out(out).squeeze(-1) # (B, T)
        
        if self.ablation_mode == "no_forecast":
            return sepsis_logits
            
        forecast_preds = self.forecast_head(out) # (B, T, 5)
        return sepsis_logits, forecast_preds


if __name__ == "__main__":
    B, T = 2, 48
    x_triplet = torch.randn(B, T, 102)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 40:] = True
    
    model = SepsisHybridModel()
    logits, forecasts = model(x_triplet, padding_mask=mask)
    print(f"Logits Shape: {logits.shape}")          # (2, 48)
    print(f"Forecasts Shape: {forecasts.shape}")    # (2, 48, 5)
