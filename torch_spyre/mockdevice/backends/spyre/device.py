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
        """
        Load SDSC JSON artifact for mock device validation.
        
        Looks for sdsc_*.json files in the provided directory.
        """
        # If already a MockOpSpec, return as-is
        if isinstance(artifact, MockOpSpec):
            logger.info(f"[FLOW] Using provided MockOpSpec: {artifact.op_spec_name}")
            return artifact
        
        # Load from file or directory path
        if isinstance(artifact, str):
            artifact_path = Path(artifact)
            
            # If it's a directory, look for sdsc_*.json files
            if artifact_path.is_dir():
                sdsc_files = list(artifact_path.glob("sdsc_*.json"))
                if not sdsc_files:
                    logger.error(f"[FLOW] No SDSC JSON files (sdsc_*.json) found in {artifact_path}")
                    raise FileNotFoundError(f"No SDSC JSON files (sdsc_*.json) found in {artifact_path}")
                
                # Use the first SDSC file (typically sdsc_0.json)
                json_path = sorted(sdsc_files)[0]
                logger.info(f"[FLOW] Found SDSC JSON: {json_path.name}")
                artifact_path = json_path
            
            # Load the SDSC JSON file
            with open(artifact_path) as f:
                data = json.load(f)
            from ...core.serialization import SDSCSerializer
            logger.info(f"[FLOW] Loading SDSC artifact: {artifact_path.name}")
            return SDSCSerializer.from_dict(data)
        
        # Parse from dict (SDSC format)
        if isinstance(artifact, dict):
            from ...core.serialization import SDSCSerializer
            return SDSCSerializer.from_dict(artifact)
        
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


