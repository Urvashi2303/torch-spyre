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


class SDSCSerializer:
    """
    Parse/deserialize SDSC JSON format to MockOpSpec.
    
    SDSC JSON is the hardware compilation format generated by superdsc.py.
    This serializer extracts the necessary information to create a MockOpSpec
    for validation and mock execution.
    """
    
    @staticmethod
    def from_dict(sdsc_data: dict) -> MockOpSpec:
        """
        Parse SDSC JSON structure and convert to MockOpSpec.
        
        SDSC Structure:
        {
          "operation_name": {
            "numCoresUsed_": 1,
            "dscs_": [{
              "operation_name": {
                "N_": {"name_": "n", "dim1_": size1, ...},
                "labeledDs_": [{
                  "ldsIdx_": 0,
                  "dsName_": "Tensor0",
                  "dsType_": "layout_label",
                  "scale_": [1, 1, ...],
                  "dataFormat_": "SEN169_FP16",
                  ...
                }],
                "computeOp_": [{
                  "opFuncName": "add",
                  "exUnit": "sfp",
                  "inputLabeledDs": ["Tensor0-idx0", ...],
                  "outputLabeledDs": ["Tensor1-idx1"],
                  "attributes_": {...}
                }],
                "primaryDsInfo_": {
                  "layout_label": {
                    "layoutDimOrder_": ["dim1", "dim2"],
                    "stickDimOrder_": ["dim2"],
                    "stickSize_": [64]
                  }
                },
                ...
              }
            }]
          }
        }
        
        Args:
            sdsc_data: Dictionary containing SDSC JSON structure
            
        Returns:
            MockOpSpec object for validation and execution
            
        Raises:
            ValueError: If SDSC structure is invalid or missing required fields
        """
        logger.stage("PARSE_SDSC", "Parsing SDSC JSON to MockOpSpec")
        
        # Extract operation name (top-level key)
        if not sdsc_data:
            raise ValueError("SDSC data is empty")
        
        op_name = list(sdsc_data.keys())[0]
        logger.debug(f"[FLOW] Parsing SDSC for operation: {op_name}")
        
        sdsc_root = sdsc_data[op_name]
        
        # Navigate to DSC content
        if "dscs_" not in sdsc_root or not sdsc_root["dscs_"]:
            raise ValueError(f"SDSC missing 'dscs_' array for operation '{op_name}'")
        
        dsc_content = sdsc_root["dscs_"][0][op_name]
        
        # Extract dimensions from N_
        dimensions = {}
        if "N_" in dsc_content:
            dimensions = {
                k.rstrip("_"): v
                for k, v in dsc_content["N_"].items()
                if k != "name_"
            }
            logger.debug(f"[FLOW] Extracted dimensions: {dimensions}")
        
        # Extract layout information
        layouts = {}
        if "primaryDsInfo_" in dsc_content:
            for label, layout_info in dsc_content["primaryDsInfo_"].items():
                layouts[label] = {
                    "dim_order": layout_info.get("layoutDimOrder_", []),
                    "stick_dim_order": layout_info.get("stickDimOrder_", [None])[0],
                    "stick_size": layout_info.get("stickSize_", [64])[0]
                }
            logger.debug(f"[FLOW] Extracted {len(layouts)} layout(s)")
        
        # Extract tensors from labeledDs_
        tensors: dict[str, TensorDescriptor] = {}
        labeled_ds = dsc_content.get("labeledDs_", [])
        
        dtype_map = {
            "SEN169_FP16": torch.float16,
            "SEN169_BF16": torch.bfloat16,
            "SEN169_FP32": torch.float32,
            "SEN169_INT32": torch.int32,
        }
        
        for lds in labeled_ds:
            tensor_name = lds["dsName_"]
            lds_idx = lds["ldsIdx_"]
            layout_label = lds.get("dsType_", "")
            scale = lds.get("scale_", [])
            data_format = lds.get("dataFormat_", "SEN169_FP32")
            
            # Determine dtype
            dtype = dtype_map.get(data_format, torch.float32)
            
            # Get layout info to determine shape
            layout_info = layouts.get(layout_label, {})
            dim_order = layout_info.get("dim_order", [])
            stick_size = layout_info.get("stick_size", 64)
            
            # Build device_size from dimensions and layout
            device_size = []
            for dim_name in dim_order:
                if dim_name in dimensions:
                    device_size.append(dimensions[dim_name])
                else:
                    # Stick dimension might not be in N_
                    device_size.append(stick_size)
            
            # Determine roles based on scale (heuristic)
            # Typically: inputs have positive scales, outputs may have mixed
            # We'll refine this when we parse computeOp_
            roles = set()
            
            stride = [1] * len(device_size)
            dim_map = list(range(len(device_size)))
            
            tensors[tensor_name] = TensorDescriptor(
                name=tensor_name,
                roles=roles,  # Will be set when parsing compute ops
                shape=device_size,
                stride=stride,
                dtype=dtype,
                device_size=device_size,
                dim_map=dim_map,
                device_dtype=data_format,
                metadata={
                    "ldsIdx": lds_idx,
                    "layout": layout_label,
                    "scale": scale,
                },
            )
        
        logger.debug(f"[FLOW] Extracted {len(tensors)} tensor(s)")
        
        # Extract compute operations from computeOp_
        compute_ops = []
        compute_op_list = dsc_content.get("computeOp_", [])
        
        num_cores = sdsc_root.get("numCoresUsed_", 1)
        
        for cop in compute_op_list:
            op_func_name = cop.get("opFuncName", op_name)
            ex_unit = cop.get("exUnit", "sfp")
            
            # Parse input tensor references
            input_refs = cop.get("inputLabeledDs", [])
            input_tensor_names = []
            for ref in input_refs:
                # Format: "Tensor0-idx0"
                tensor_name = ref.split("-")[0]
                input_tensor_names.append(tensor_name)
                # Mark as INPUT
                if tensor_name in tensors:
                    tensors[tensor_name].roles.add(INPUT)
            
            # Parse output tensor references
            output_refs = cop.get("outputLabeledDs", [])
            output_tensor_names = []
            for ref in output_refs:
                tensor_name = ref.split("-")[0]
                output_tensor_names.append(tensor_name)
                # Mark as OUTPUT
                if tensor_name in tensors:
                    tensors[tensor_name].roles.add(OUTPUT)
            
            # Extract attributes
            attributes = cop.get("attributes_", {})
            
            compute_op = ComputeOp(
                op_func_name=op_func_name,
                ex_unit=ex_unit,
                input_tensor_names=input_tensor_names,
                output_tensor_names=output_tensor_names,
                num_cores=num_cores,
                attributes=attributes,
            )
            compute_ops.append(compute_op)
        
        logger.debug(f"[FLOW] Extracted {len(compute_ops)} compute operation(s)")
        
        # Build metadata
        metadata = {
            "num_cores": num_cores,
            "sdsc_format": True,
            "layouts": layouts,
        }
        
        # Add coordinate masking if present
        if "coordinateMasking_" in dsc_content:
            metadata["coordinate_masking"] = dsc_content["coordinateMasking_"]
        
        # Add iteration space info
        if "numWkSlicesPerDim_" in sdsc_root:
            metadata["work_slices"] = sdsc_root["numWkSlicesPerDim_"]
        
        mock_spec = MockOpSpec(
            op_spec_name=op_name,
            dimensions=dimensions,
            tensors=tensors,
            compute_ops=compute_ops,
            metadata=metadata,
        )
        
        logger.info(f"[FLOW] Successfully parsed SDSC to MockOpSpec: {op_name}")
        
        # Print detailed MockOpSpec structure
        print("\n mock_spec: " , mock_spec)
        print("="*70 + "\n")
        
        return mock_spec
    
    @staticmethod
    def from_file(path: str | Path) -> MockOpSpec:
        """
        Load SDSC JSON from file and parse to MockOpSpec.
        
        Args:
            path: Path to SDSC JSON file (e.g., sdsc_0.json)
            
        Returns:
            MockOpSpec object
        """
        path_obj = Path(path)
        if not path_obj.exists():
            logger.error(f"[FLOW] SDSC file not found: {path}")
            raise FileNotFoundError(f"SDSC file not found: {path}")
        
        logger.info(f"[FLOW] Loading SDSC from: {path_obj.name}")
        
        with open(path_obj) as f:
            sdsc_data = json.load(f)
        
        return SDSCSerializer.from_dict(sdsc_data)


def load_sdsc(path: str | Path) -> MockOpSpec:
    """Convenience function to load SDSC JSON."""
    return SDSCSerializer.from_file(path)
