"""
torch_spyre.mockdevice.backends.spyre.layout
--------------------------------------------
SpyreStickLayout
"""
from __future__ import annotations
import torch
from ...core import AbstractLayoutTransformer, TensorDescriptor
from ...logger import get_logger

logger = get_logger("SpyreLayout")


class SpyreStickLayout(AbstractLayoutTransformer):
    """Spyre stick-layout transformer"""

    def to_contiguous(self, tensor: torch.Tensor, td: TensorDescriptor) -> torch.Tensor:
        """Strip stick-alignment padding and return a C-contiguous tensor."""
        logger.info(f"[FLOW] SpyreStickLayout.to_contiguous() for tensor '{td.name}'")
        logger.debug(f"[FLOW] Tensor shape: {list(tensor.shape)}, target shape: {td.shape}")
        return tensor.contiguous()

    def from_contiguous(self, tensor: torch.Tensor, td: TensorDescriptor) -> torch.Tensor:
        """No-op: no physical HBM to write to in mock mode."""
        logger.info(f"[FLOW] SpyreStickLayout.from_contiguous() for tensor '{td.name}' (no-op in mock)")
        return tensor
