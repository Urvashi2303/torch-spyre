"""
torch_spyre.mockdevice.core.serialization
-----------------------------------------
OpSpec serialization/deserialization
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch
from .graph import MockOpSpec, TensorDescriptor, ComputeOp
from ..logger import get_logger

logger = get_logger("Serialization")


class OpSpecSerializer:
    """Serialize/deserialize MockOpSpec to/from JSON"""
    
    @staticmethod
    def save(spec: MockOpSpec, path: str | Path) -> None:
        """Save MockOpSpec to JSON file."""
        logger.stage("SAVE_OPSPEC", f"Saving OpSpec to {path}")
        # Serialization logic would go here - removed for flow-only version
        path_obj = Path(path)
    
    @staticmethod
    def load(path: str | Path) -> MockOpSpec:
        """Load MockOpSpec from JSON file."""
        logger.stage("LOAD_OPSPEC", f"Loading OpSpec from {path}")
        
        path_obj = Path(path)
        if not path_obj.exists():
            logger.error(f"[FLOW] OpSpec file not found: {path}")
            raise FileNotFoundError(f"OpSpec file not found: {path}")
        
        # Deserialization logic would go here - removed for flow-only version
        # Create minimal OpSpec for flow demonstration
        spec = MockOpSpec(
            op_spec_name="loaded_op",
            dimensions={},
            tensors={},
            compute_ops=[],
            metadata={},
        )
        
        return spec
    
    @staticmethod
    def to_dict(spec: MockOpSpec) -> dict[str, Any]:
        """Convert MockOpSpec to dict"""
        # Conversion logic would go here - removed for flow-only version
        return {
            "op_spec_name": spec.op_spec_name,
            "dimensions": spec.dimensions,
            "tensors": {},
            "compute_ops": [],
            "metadata": spec.metadata,
        }
    
    @staticmethod
    def from_dict(data: dict[str, Any]) -> MockOpSpec:
        """Create MockOpSpec from dict"""
        # Extract op_spec_name from various possible locations in the JSON structure
        op_spec_name = "unknown"
        
        # Check if op_spec_name is directly in data
        if "op_spec_name" in data:
            op_spec_name = data["op_spec_name"]
        # Check if there's a specs array with op field (mock_op_specs.json format)
        elif "specs" in data and len(data["specs"]) > 0:
            first_spec = data["specs"][0]
            if "op" in first_spec:
                op_spec_name = first_spec["op"]
        # Check if kernel_name is present (alternative format)
        elif "kernel_name" in data:
            op_spec_name = data["kernel_name"]
        
        logger.debug(f"[FLOW] Extracted op_spec_name: {op_spec_name}")
        
        return MockOpSpec(
            op_spec_name=op_spec_name,
            dimensions=data.get("dimensions", {}),
            tensors={},
            compute_ops=[],
            metadata=data.get("metadata", {}),
        )


def save_opspec(spec: MockOpSpec, path: str | Path) -> None:
    """Convenience function to save OpSpec."""
    OpSpecSerializer.save(spec, path)


def load_opspec(path: str | Path) -> MockOpSpec:
    """Convenience function to load OpSpec."""
    return OpSpecSerializer.load(path)


