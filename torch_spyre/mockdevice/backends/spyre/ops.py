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
    return inputs

logger.info(f"[FLOW] Registered {len(SPYRE_OPS.override_ops)} Spyre op overrides: {sorted(SPYRE_OPS.override_ops)}")
