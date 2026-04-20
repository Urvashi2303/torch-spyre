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
        # If already a MockOpSpec, return as-is
        if isinstance(artifact, MockOpSpec):
            logger.info(f"[FLOW] Using provided MockOpSpec: {artifact.op_spec_name}")
            return artifact
        
        # Load from file or directory path
        if isinstance(artifact, str):
            artifact_path = Path(artifact)
            
            # If it's a directory, look for mock_op_specs.json inside
            if artifact_path.is_dir():
                json_path = artifact_path / "mock_op_specs.json"
                print(f" Loading OpSpec from path: json_path: {json_path}")
                if not json_path.exists():
                    logger.error(f"[FLOW] mock_op_specs.json not found in {artifact_path}")
                    raise FileNotFoundError(f"mock_op_specs.json not found in {artifact_path}")
                artifact_path = json_path
            
            # Load the JSON file
            with open(artifact_path) as f:
                data = json.load(f)
            from ...core.serialization import OpSpecSerializer
            logger.info(f"[FLOW] Loading OpSpec artifact: {artifact_path.name}")
            return OpSpecSerializer.from_dict(data)
        
        # Parse from dict
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


