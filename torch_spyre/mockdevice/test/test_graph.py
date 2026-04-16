"""
Unit tests for torch_spyre.mockdevice.core.graph dataclasses.

Tests cover:
- Instantiation of TensorDescriptor, ComputeOp, and MockOpSpec
- Field access for all dataclasses
- Mutable default isolation (ensuring default_factory creates independent instances)
"""

import pytest
import torch
from torch_spyre.mockdevice.core.graph import (
    TensorDescriptor,
    ComputeOp,
    MockOpSpec,
    INPUT,
    OUTPUT,
    ROLES,
)


class TestTensorDescriptor:
    """Test suite for TensorDescriptor dataclass."""

    def test_instantiation_basic(self):
        """Test basic instantiation with all required fields."""
        td = TensorDescriptor(
            name="tensor1",
            roles={INPUT},
            shape=[512, 1024],
            stride=[1024, 1],
            dtype=torch.float16,
            device_size=[16, 512, 64],
            dim_map=[1, 0, 1],
            device_dtype="SEN169_FP16",
        )
        
        assert td.name == "tensor1"
        assert td.roles == {INPUT}
        assert td.shape == [512, 1024]
        assert td.metadata == {}

    def test_instantiation_with_metadata(self):
        """Test instantiation with custom metadata."""
        metadata = {"custom_key": "custom_value", "version": 1}
        td = TensorDescriptor(
            name="tensor2",
            roles={OUTPUT},
            shape=[256],
            stride=[1],
            dtype=torch.float32,
            device_size=[256],
            dim_map=[0],
            device_dtype="FP32",
            metadata=metadata,
        )
        
        assert td.metadata == {"custom_key": "custom_value", "version": 1}

    def test_field_access_all_fields(self):
        """Test accessing all fields of TensorDescriptor."""
        td = TensorDescriptor(
            name="test_tensor",
            roles={INPUT, OUTPUT},
            shape=[128, 256, 512],
            stride=[131072, 512, 1],
            dtype=torch.bfloat16,
            device_size=[8, 128, 32],
            dim_map=[2, 1, 0],
            device_dtype="BF16",
        )
        
        # Access all fields
        assert isinstance(td.roles, set)
        assert isinstance(td.shape, list)
        assert isinstance(td.dtype, torch.dtype)
        assert isinstance(td.device_dtype, str)
        assert isinstance(td.metadata, dict)

    def test_mutable_default_isolation_metadata(self):
        """Test that metadata default_factory creates independent dict instances."""
        td1 = TensorDescriptor(
            name="tensor1",
            roles={INPUT},
            shape=[10],
            stride=[1],
            dtype=torch.float32,
            device_size=[10],
            dim_map=[0],
            device_dtype="FP32",
        )
        
        td2 = TensorDescriptor(
            name="tensor2",
            roles={OUTPUT},
            shape=[20],
            stride=[1],
            dtype=torch.float32,
            device_size=[20],
            dim_map=[0],
            device_dtype="FP32",
        )
        
        # Modify metadata of td1
        td1.metadata["key1"] = "value1"
        td1.metadata["shared"] = "from_td1"
        
        # Modify metadata of td2
        td2.metadata["key2"] = "value2"
        td2.metadata["shared"] = "from_td2"
        
        # Verify isolation
        assert "key1" in td1.metadata
        assert "key1" not in td2.metadata
        assert "key2" in td2.metadata
        assert "key2" not in td1.metadata
        assert td1.metadata["shared"] == "from_td1"
        assert td2.metadata["shared"] == "from_td2"

    def test_is_input_output_method(self):
        """Test is_input() method."""
        td_input = TensorDescriptor(
            name="input_tensor",
            roles={INPUT},
            shape=[10],
            stride=[1],
            dtype=torch.float32,
            device_size=[10],
            dim_map=[0],
            device_dtype="FP32",
        )
        
        td_output = TensorDescriptor(
            name="output_tensor",
            roles={OUTPUT},
            shape=[10],
            stride=[1],
            dtype=torch.float32,
            device_size=[10],
            dim_map=[0],
            device_dtype="FP32",
        )
        
        td_inplace = TensorDescriptor(
            name="inplace_tensor",
            roles={INPUT, OUTPUT},
            shape=[10],
            stride=[1],
            dtype=torch.float32,
            device_size=[10],
            dim_map=[0],
            device_dtype="FP32",
        )
        
        assert td_input.is_input() is True
        assert td_input.is_output() is False
        assert td_input.is_inplace() is False
        assert td_output.is_input() is False
        assert td_output.is_output() is True
        assert td_output.is_inplace() is False
        assert td_inplace.is_input() is True
        assert td_inplace.is_output() is True
        assert td_inplace.is_inplace() is True


class TestComputeOp:
    """Test suite for ComputeOp dataclass."""

    def test_instantiation_basic(self):
        """Test basic instantiation with all required fields."""
        op = ComputeOp(
            op_func_name="matmul",
            ex_unit="VXM",
            input_tensor_names=["input1", "input2"],
            output_tensor_names=["output1"],
            num_cores=16,
        )
        
        assert op.op_func_name == "matmul"
        assert op.input_tensor_names == ["input1", "input2"]
        assert op.num_cores == 16
        assert op.attributes == {}

    def test_instantiation_with_attributes(self):
        """Test instantiation with custom attributes."""
        attributes = {"alpha": 1.0, "beta": 0.5, "transpose_a": True}
        op = ComputeOp(
            op_func_name="gemm",
            ex_unit="VXM",
            input_tensor_names=["A", "B"],
            output_tensor_names=["C"],
            num_cores=32,
            attributes=attributes,
        )
        
        assert op.attributes == {"alpha": 1.0, "beta": 0.5, "transpose_a": True}

    def test_field_access_all_fields(self):
        """Test accessing all fields of ComputeOp."""
        op = ComputeOp(
            op_func_name="add",
            ex_unit="MXM",
            input_tensor_names=["x", "y"],
            output_tensor_names=["z"],
            num_cores=8,
        )
        
        # Access all fields
        assert isinstance(op.ex_unit, str)
        assert isinstance(op.output_tensor_names, list)
        assert isinstance(op.num_cores, int)
        assert isinstance(op.attributes, dict)

    def test_mutable_default_isolation_attributes(self):
        """Test that attributes default_factory creates independent dict instances."""
        op1 = ComputeOp(
            op_func_name="op1",
            ex_unit="VXM",
            input_tensor_names=["in1"],
            output_tensor_names=["out1"],
            num_cores=4,
        )
        
        op2 = ComputeOp(
            op_func_name="op2",
            ex_unit="MXM",
            input_tensor_names=["in2"],
            output_tensor_names=["out2"],
            num_cores=8,
        )
        
        # Modify attributes of op1
        op1.attributes["param1"] = 100
        op1.attributes["shared"] = "from_op1"
        
        # Modify attributes of op2
        op2.attributes["param2"] = 200
        op2.attributes["shared"] = "from_op2"
        
        # Verify isolation
        assert "param1" in op1.attributes
        assert "param1" not in op2.attributes
        assert "param2" in op2.attributes
        assert "param2" not in op1.attributes
        assert op1.attributes["shared"] == "from_op1"
        assert op2.attributes["shared"] == "from_op2"

    def test_resolve_tensor_names(self):
        """Test resolve_input_name and resolve_output_name methods."""
        op = ComputeOp(
            op_func_name="test_op",
            ex_unit="VXM",
            input_tensor_names=["tensor1-slice0", "tensor2-slice1"],
            output_tensor_names=["output1-slice0", "output2-slice1"],
            num_cores=4,
        )
        
        # Test input name resolution
        assert op.resolve_input_name("tensor1-slice0") == "tensor1"
        assert op.resolve_input_name("simple_name") == "simple_name"
        
        # Test output name resolution
        assert op.resolve_output_name("output1-slice0") == "output1"
        assert op.resolve_output_name("simple_name") == "simple_name"

    def test_tensor_names_properties(self):
        """Test input_names and output_names properties."""
        op = ComputeOp(
            op_func_name="test_op",
            ex_unit="VXM",
            input_tensor_names=["tensor1-slice0", "tensor2-slice1", "tensor3"],
            output_tensor_names=["out1-slice0", "out2-slice1", "out3"],
            num_cores=4,
        )
        
        # Test input_names property
        assert op.input_names == ["tensor1", "tensor2", "tensor3"]
        
        # Test output_names property
        assert op.output_names == ["out1", "out2", "out3"]


class TestMockOpSpec:
    """Test suite for MockOpSpec dataclass."""

    def test_instantiation_basic(self):
        """Test basic instantiation with all required fields."""
        tensor1 = TensorDescriptor(
            name="input",
            roles={INPUT},
            shape=[10],
            stride=[1],
            dtype=torch.float32,
            device_size=[10],
            dim_map=[0],
            device_dtype="FP32",
        )
        
        tensor2 = TensorDescriptor(
            name="output",
            roles={OUTPUT},
            shape=[10],
            stride=[1],
            dtype=torch.float32,
            device_size=[10],
            dim_map=[0],
            device_dtype="FP32",
        )
        
        op = ComputeOp(
            op_func_name="relu",
            ex_unit="VXM",
            input_tensor_names=["input"],
            output_tensor_names=["output"],
            num_cores=4,
        )
        
        spec = MockOpSpec(
            op_spec_name="relu_op",
            dimensions={"N": 10},
            tensors={"input": tensor1, "output": tensor2},
            compute_ops=[op],
        )
        
        assert spec.op_spec_name == "relu_op"
        assert spec.dimensions == {"N": 10}
        assert len(spec.tensors) == 2
        assert len(spec.compute_ops) == 1
        assert spec.metadata == {}

    def test_instantiation_with_metadata(self):
        """Test instantiation with custom metadata."""
        metadata = {"version": "1.0", "author": "test"}
        spec = MockOpSpec(
            op_spec_name="test_spec",
            dimensions={},
            tensors={},
            compute_ops=[],
            metadata=metadata,
        )
        
        assert spec.metadata == {"version": "1.0", "author": "test"}

    def test_field_access_all_fields(self):
        """Test accessing all fields of MockOpSpec."""
        tensor = TensorDescriptor(
            name="t1",
            roles={INPUT},
            shape=[5],
            stride=[1],
            dtype=torch.float32,
            device_size=[5],
            dim_map=[0],
            device_dtype="FP32",
        )
        
        op = ComputeOp(
            op_func_name="test",
            ex_unit="VXM",
            input_tensor_names=["t1"],
            output_tensor_names=["t2"],
            num_cores=2,
        )
        
        spec = MockOpSpec(
            op_spec_name="test_spec",
            dimensions={"D": 5},
            tensors={"t1": tensor},
            compute_ops=[op],
        )
        
        # Access all fields
        assert isinstance(spec.op_spec_name, str)
        assert isinstance(spec.dimensions, dict)
        assert isinstance(spec.compute_ops, list)
        assert isinstance(spec.metadata, dict)

    def test_mutable_default_isolation_metadata(self):
        """Test that metadata default_factory creates independent dict instances."""
        spec1 = MockOpSpec(
            op_spec_name="spec1",
            dimensions={},
            tensors={},
            compute_ops=[],
        )
        
        spec2 = MockOpSpec(
            op_spec_name="spec2",
            dimensions={},
            tensors={},
            compute_ops=[],
        )
        
        # Modify metadata of spec1
        spec1.metadata["key1"] = "value1"
        spec1.metadata["shared"] = "from_spec1"
        
        # Modify metadata of spec2
        spec2.metadata["key2"] = "value2"
        spec2.metadata["shared"] = "from_spec2"
        
        # Verify isolation
        assert "key1" in spec1.metadata
        assert "key1" not in spec2.metadata
        assert "key2" in spec2.metadata
        assert "key2" not in spec1.metadata
        assert spec1.metadata["shared"] == "from_spec1"
        assert spec2.metadata["shared"] == "from_spec2"

    def test_tensor_filtering_properties(self):
        """Test input_tensors and output_tensors properties."""
        input_tensor = TensorDescriptor(
            name="input1",
            roles={INPUT},
            shape=[10],
            stride=[1],
            dtype=torch.float32,
            device_size=[10],
            dim_map=[0],
            device_dtype="FP32",
        )
        
        output_tensor = TensorDescriptor(
            name="output1",
            roles={OUTPUT},
            shape=[10],
            stride=[1],
            dtype=torch.float32,
            device_size=[10],
            dim_map=[0],
            device_dtype="FP32",
        )
        
        inplace_tensor = TensorDescriptor(
            name="inplace1",
            roles={INPUT, OUTPUT},
            shape=[10],
            stride=[1],
            dtype=torch.float32,
            device_size=[10],
            dim_map=[0],
            device_dtype="FP32",
        )
        
        spec = MockOpSpec(
            op_spec_name="test_spec",
            dimensions={},
            tensors={
                "input1": input_tensor,
                "output1": output_tensor,
                "inplace1": inplace_tensor,
            },
            compute_ops=[],
        )
        
        # Test input_tensors property
        input_tensors = spec.input_tensors
        assert len(input_tensors) == 2
        assert "input1" in input_tensors
        assert "inplace1" in input_tensors
        assert "output1" not in input_tensors
        
        # Test output_tensors property
        output_tensors = spec.output_tensors
        assert len(output_tensors) == 2
        assert "output1" in output_tensors
        assert "inplace1" in output_tensors
        assert "input1" not in output_tensors
