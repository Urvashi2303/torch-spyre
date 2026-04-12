"""
torch_spyre.mockdevice.backends.spyre.device
--------------------------------------------
MockSpyreDevice
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch
from ...core import AbstractMockDevice, MockOpDispatcher, MockOpSpec
from ...logger import get_logger
from .validator import MockOpSpecValidator
from .layout import SpyreStickLayout
from .ops import SPYRE_OPS

logger = get_logger("SpyreDevice")

SPYRE_ATTR_MAP = {
    "fidelity_": "approximate",
}

class MockSpyreDevice(AbstractMockDevice):
    """Software-only mock of the Spyre accelerator"""

    @property
    def device_name(self): 
        return "Spyre"

    def _make_validator(self):
        return MockOpSpecValidator()
    
    def _make_dispatcher(self):
        return MockOpDispatcher(
            registry=SPYRE_OPS,
            layout=SpyreStickLayout(),
            attr_map=SPYRE_ATTR_MAP,
            precision_dtype=torch.float32,
            verbose=self.verbose,
        )

    def _load_artifact(self, artifact):
        """Load OpSpec JSON artifact (mock_op_specs.json format only)."""
        if isinstance(artifact, str):
            artifact_path = Path(artifact)
            with open(artifact) as f:
                data = json.load(f)
            from ...core.serialization import OpSpecSerializer
            logger.info(f"[FLOW] Loading OpSpec artifact: {artifact_path.name}")
            return OpSpecSerializer.from_dict(data)
        if isinstance(artifact, MockOpSpec):
            return artifact
        if isinstance(artifact, dict):
            from ...core.serialization import OpSpecSerializer
            return OpSpecSerializer.from_dict(artifact)
        logger.error(f"[FLOW] Invalid artifact type: {type(artifact).__name__}")
        raise TypeError(f"artifact must be a file path (str), parsed dict, or MockOpSpec")

    def _on_submit_start(self, spec: MockOpSpec, artifact: Any, **kwargs: Any) -> None:
        """Emit Spyre runtime simulation log lines."""
        artifact_path = artifact if isinstance(artifact, str) else "<in-memory>"
        
        print(f"RUN: {spec.op_spec_name} {artifact_path}")
        
        if self.verbose:
            dims  = spec.dimensions
            cores = spec.metadata.get("num_cores", "?")
            logger.info(f"[FLOW] Spyre kernel: op={spec.op_spec_name}, dims={dims}, cores={cores}")


