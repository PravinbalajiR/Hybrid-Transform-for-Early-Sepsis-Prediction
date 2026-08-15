"""
transformer_encoder.py
----------------------
Alias module pointing to TACTModel in tact_model.py for backward compatibility.
"""

from models.transformer.tact_model import TACTModel, SepsisTransformer

__all__ = ["TACTModel", "SepsisTransformer"]
