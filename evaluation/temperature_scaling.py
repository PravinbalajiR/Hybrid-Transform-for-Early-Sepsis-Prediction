"""
temperature_scaling.py
----------------------
Implements post-hoc temperature scaling to improve model calibration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import numpy as np


class TemperatureScaler(nn.Module):
    """
    A thin decorator that wraps a model with temperature scaling.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x, padding_mask=None):
        # We assume the model outputs logits
        logits = self.model(x, padding_mask=padding_mask)
        
        # If model returns tuple (e.g. logits, forecast)
        if isinstance(logits, tuple):
            return logits[0] / self.temperature, logits[1]
            
        return logits / self.temperature


def optimize_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Finds the optimal temperature T to minimize NLL (Cross Entropy) on the validation set.
    """
    nll_criterion = nn.BCEWithLogitsLoss()
    
    # Initial NLL
    before_temperature_nll = nll_criterion(logits, labels).item()
    print(f"Before Temperature - NLL: {before_temperature_nll:.4f}")
    
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    # L-BFGS optimizer
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss
        
    optimizer.step(eval)
    
    after_temperature_nll = nll_criterion(logits / temperature, labels).item()
    print(f"Optimal Temperature: {temperature.item():.4f}")
    print(f"After Temperature  - NLL: {after_temperature_nll:.4f}")
    
    return temperature.item()
