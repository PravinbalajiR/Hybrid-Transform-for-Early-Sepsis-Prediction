import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.losses import FocalLoss, UtilityAwareLoss

def test_loss_equivalence():
    print("Running Sanity Checks for Custom Losses...")
    
    logits = torch.randn(2, 500)
    targets = torch.randint(0, 2, (2, 500)).float()
    
    # 1. Test FocalLoss equivalence with gamma=0
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([47.66]))
    focal = FocalLoss(pos_weight=47.66, gamma=0.0)
    
    loss_bce = bce(logits, targets)
    loss_focal = focal(logits, targets)
    
    print(f"BCE Loss: {loss_bce.item():.6f}")
    print(f"Focal Loss (gamma=0): {loss_focal.item():.6f}")
    
    diff = abs(loss_bce.item() - loss_focal.item())
    print(f"Difference: {diff:.6e}")
    if diff < 1e-5:
        print("[PASS] FocalLoss with gamma=0 reduces to BCEWithLogitsLoss.")
    else:
        print("[FAIL] FocalLoss equivalence test failed.")
        
    # 2. Test valid_mask application
    valid_mask = torch.ones_like(targets, dtype=torch.bool)
    valid_mask[:, 250:] = False # Pad half the sequence
    
    logits_padded = logits.clone()
    targets_padded = targets.clone()
    logits_padded[:, 250:] = -10.0 # Easy negative predictions
    targets_padded[:, 250:] = 0.0
    
    focal_unmasked = focal(logits_padded, targets_padded)
    focal_masked = focal(logits_padded[valid_mask], targets_padded[valid_mask])
    
    print(f"Focal Unmasked: {focal_unmasked.item():.6f}")
    print(f"Focal Masked: {focal_masked.item():.6f}")
    if focal_unmasked < focal_masked:
        print("[PASS] Padding dilution successfully observed (Unmasked is smaller).")
    else:
        print("[FAIL] Masking logic failed.")
        
    util = UtilityAwareLoss(base_pos_weight=47.66)
    util_unmasked = util(logits_padded, targets_padded, valid_mask=None)
    util_masked = util(logits_padded, targets_padded, valid_mask=valid_mask)
    
    print(f"Utility Unmasked: {util_unmasked.item():.6f}")
    print(f"Utility Masked: {util_masked.item():.6f}")
    if util_unmasked < util_masked:
        print("[PASS] Padding dilution in UtilityAwareLoss successfully observed and handled.")
    else:
        print("[FAIL] UtilityAwareLoss masking logic failed.")

if __name__ == "__main__":
    test_loss_equivalence()
