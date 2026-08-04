"""
losses.py
---------
Custom loss functions for the TACT-UGO architecture.
Includes Focal Loss for class imbalance and Utility-Aware BCE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Binary Focal Loss with pos_weight handling.
    """
    def __init__(self, pos_weight: float = 47.66, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight_t = torch.tensor([self.pos_weight], device=logits.device)
        
        # Get unweighted BCE for pt calculation
        unweighted_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-unweighted_bce) 
        
        # Get weighted BCE for the base loss
        weighted_bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pos_weight_t
        )
        
        focal_loss = (1 - pt) ** self.gamma * weighted_bce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class UtilityAwareLoss(nn.Module):
    """
    Utility-Aware BCE Loss.
    Scales the loss for positive examples based on their temporal distance to sepsis onset.
    Weights peak at t_sepsis (15.0) and decrease backwards.
    Takes an optional valid_mask to avoid averaging over padded tokens.
    """
    def __init__(self, base_pos_weight: float = 47.66, reduction: str = "mean"):
        super().__init__()
        self.base_pos_weight = base_pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        B, T = logits.shape
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        weights = torch.ones_like(targets)
        
        for b in range(B):
            if targets[b].sum() > 0:
                pos_indices = (targets[b] == 1).nonzero(as_tuple=True)[0]
                
                # Apply the specific weights requested by the user
                utility_weights = [5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
                
                for i, idx in enumerate(pos_indices):
                    if i < len(utility_weights):
                        weights[b, idx] = utility_weights[i]
                    else:
                        weights[b, idx] = 15.0 # Max weight for hours > t0
        
        weighted_loss = bce_loss * weights
        
        if valid_mask is not None:
            weighted_loss = weighted_loss[valid_mask]
        
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss
