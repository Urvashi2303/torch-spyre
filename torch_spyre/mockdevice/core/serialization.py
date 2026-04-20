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
from .graph import INPUT, OUTPUT, MockOpSpec, TensorDescriptor, ComputeOp
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
        spec_data = data
        metadata = dict(data.get("metadata", {}))

        if "specs" in data:
            specs = data.get("specs", [])
            if not specs:
                raise ValueError("mock_op_specs.json contains an empty 'specs' array")
            spec_data = specs[0]
            metadata.setdefault("kernel_name", data.get("kernel_name"))
            metadata.setdefault("num_specs", data.get("num_specs"))

        op_spec_name = spec_data.get("op_spec_name") or spec_data.get("op") or data.get("kernel_name", "unknown")
        logger.debug(f"[FLOW] Extracted op_spec_name: {op_spec_name}")

        iteration_space = spec_data.get("iteration_space", {})
        dimensions = spec_data.get("dimensions")
        if dimensions is None:
            dimensions = {
                symbol: axis_info.get("range")
                for symbol, axis_info in iteration_space.items()
            }

        args = spec_data.get("args", [])
        tensors: dict[str, TensorDescriptor] = {}
        input_tensor_names: list[str] = []
        output_tensor_names: list[str] = []

        dtype_map = {
            "DataFormats.SEN169_FP16": torch.float16,
            "DataFormats.SEN169_BF16": torch.bfloat16,
            "DataFormats.SEN169_FP32": torch.float32,
            "DataFormats.SEN169_INT32": torch.int32,
        }

        for position, arg in enumerate(args):
            tensor_name = f"Tensor{position}"
            roles = {INPUT} if arg.get("is_input", False) else {OUTPUT}
            device_size = list(arg.get("device_size", []))
            stride = [1] * len(device_size)
            dim_map = list(range(len(device_size)))
            device_dtype = arg.get("device_dtype", "unknown")
            dtype = dtype_map.get(device_dtype, torch.float32)

            tensors[tensor_name] = TensorDescriptor(
                name=tensor_name,
                roles=roles,
                shape=device_size,
                stride=stride,
                dtype=dtype,
                device_size=device_size,
                dim_map=dim_map,
                device_dtype=device_dtype,
                metadata={
                    "arg_index": arg.get("arg_index"),
                    "device_coordinates": arg.get("device_coordinates", []),
                    "allocation": arg.get("allocation"),
                },
            )

            if INPUT in roles:
                input_tensor_names.append(tensor_name)
            if OUTPUT in roles:
                output_tensor_names.append(tensor_name)

        num_cores = 1
        if iteration_space:
            num_cores = max(
                axis_info.get("core_division", 1)
                for axis_info in iteration_space.values()
            )

        compute_op = ComputeOp(
            op_func_name=spec_data.get("op", op_spec_name),
            ex_unit="sfp",
            input_tensor_names=input_tensor_names,
            output_tensor_names=output_tensor_names,
            num_cores=num_cores,
            attributes=spec_data.get("op_info", {}),
        )

        metadata.setdefault("num_cores", num_cores)
        metadata.setdefault("iteration_space", iteration_space)

        return MockOpSpec(
            op_spec_name=op_spec_name,
            dimensions=dimensions,
            tensors=tensors,
            compute_ops=[compute_op],
            metadata=metadata,
        )


def save_opspec(spec: MockOpSpec, path: str | Path) -> None:
    """Convenience function to save OpSpec."""
    OpSpecSerializer.save(spec, path)


def load_opspec(path: str | Path) -> MockOpSpec:
    """Convenience function to load OpSpec."""
    return OpSpecSerializer.load(path)
