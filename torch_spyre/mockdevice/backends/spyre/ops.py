"""
torch_spyre.mockdevice.backends.spyre.ops
-----------------------------------------
SDSC op dispatch functions
"""
from __future__ import annotations
import torch
from ...core import OpRegistry, ComputeOp
from ...logger import get_logger

logger = get_logger("SpyreOps")

SPYRE_OPS = OpRegistry("Spyre")

@SPYRE_OPS.register("gelu")
def _dispatch_gelu(inputs: list[torch.Tensor], op: ComputeOp, verbose: bool) -> list[torch.Tensor]:
    """GELU dispatch"""
    logger.info("[FLOW] Dispatching gelu op")
    x = inputs[0]
    # GELU formula: x * 0.5 * (1 + erf(x / sqrt(2)))
    result = x * 0.5 * (1.0 + torch.erf(x / 1.41421356237))
    return [result]

@SPYRE_OPS.register("gelufwd")
def _dispatch_gelufwd(inputs: list[torch.Tensor], op: ComputeOp, verbose: bool) -> list[torch.Tensor]:
    """GELU forward dispatch (same as gelu)"""
    logger.info("[FLOW] Dispatching gelufwd op")
    x = inputs[0]
    # GELU formula: x * 0.5 * (1 + erf(x / sqrt(2)))
    result = x * 0.5 * (1.0 + torch.erf(x / 1.41421356237))
    return [result]

logger.info(f"[FLOW] Registered {len(SPYRE_OPS.override_ops)} Spyre op overrides: {sorted(SPYRE_OPS.override_ops)}")
