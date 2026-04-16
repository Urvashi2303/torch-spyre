import logging

import pytest
import torch

from torch_spyre.mockdevice.backends.spyre.validator import MockOpSpecValidator
from torch_spyre.mockdevice.core.graph import ComputeOp, MockOpSpec, TensorDescriptor

logger = logging.getLogger(__name__)


def _make_tensor(
    name: str,
    roles: set[str],
    shape: list[int],
) -> TensorDescriptor:
    return TensorDescriptor(
        name=name,
        roles=roles,
        shape=shape,
        stride=[1] * len(shape),
        dtype=torch.float16,
        device_size=shape,
        dim_map=list(range(len(shape))),
        device_dtype="DataFormats.SEN169_FP16",
    )


def test_validate_accepts_well_formed_mock_opspec():
    spec = MockOpSpec(
        op_spec_name="add",
        dimensions={"i": 100, "j": 64},
        tensors={
            "Tensor0": _make_tensor("Tensor0", {"INPUT"}, [100, 64]),
            "Tensor1": _make_tensor("Tensor1", {"INPUT"}, [100, 64]),
            "Tensor2": _make_tensor("Tensor2", {"OUTPUT"}, [100, 64]),
        },
        compute_ops=[
            ComputeOp(
                op_func_name="add",
                ex_unit="sfp",
                input_tensor_names=["Tensor0", "Tensor1"],
                output_tensor_names=["Tensor2"],
                num_cores=1,
            )
        ],
    )

    logger.info("Positive validator case: validating well-formed MockOpSpec '%s'", spec.op_spec_name)
    logger.info("Spec tensors=%s compute_ops=%s", list(spec.tensors), [op.op_func_name for op in spec.compute_ops])

    MockOpSpecValidator().validate(spec)

    logger.info("Positive validator case passed for '%s'", spec.op_spec_name)


def test_validate_reports_tensor_shape_and_role_errors():
    spec = MockOpSpec(
        op_spec_name="bad_tensor_spec",
        dimensions={"i": 100},
        tensors={
            "Tensor0": _make_tensor("Tensor0", set(), [100]),
            "Tensor1": _make_tensor("Tensor1", {"BAD_ROLE"}, [0]),
        },
        compute_ops=[
            ComputeOp(
                op_func_name="add",
                ex_unit="sfp",
                input_tensor_names=["Tensor0"],
                output_tensor_names=["Tensor1"],
                num_cores=1,
            )
        ],
    )

    logger.info("Negative validator case: expecting tensor role/shape validation errors for '%s'", spec.op_spec_name)

    with pytest.raises(ValueError) as exc_info:
        MockOpSpecValidator().validate(spec)

    message = str(exc_info.value)
    logger.error("Captured validator error for '%s':\n%s", spec.op_spec_name, message)

    assert "Tensor 'Tensor0': roles is empty" in message
    assert "Tensor 'Tensor1': unknown role(s)" in message
    assert "Tensor 'Tensor1': all shape dims must be > 0" in message
    assert "No INPUT tensor found in MockOpSpec" in message
    assert "No OUTPUT tensor found in MockOpSpec" in message


def test_validate_reports_missing_compute_ops_and_broken_wiring():
    spec = MockOpSpec(
        op_spec_name="bad_wiring_spec",
        dimensions={"i": 100},
        tensors={
            "Tensor0": _make_tensor("Tensor0", {"INPUT"}, [100]),
            "Tensor1": _make_tensor("Tensor1", {"OUTPUT"}, [100]),
        },
        compute_ops=[
            ComputeOp(
                op_func_name="",
                ex_unit="sfp",
                input_tensor_names=[],
                output_tensor_names=["MissingTensor"],
                num_cores=1,
            )
        ],
    )

    logger.info("Negative validator case: expecting compute-op and wiring validation errors for '%s'", spec.op_spec_name)

    with pytest.raises(ValueError) as exc_info:
        MockOpSpecValidator().validate(spec)

    message = str(exc_info.value)
    logger.error("Captured validator error for '%s':\n%s", spec.op_spec_name, message)

    assert "compute_ops[0]: op_func_name is empty" in message
    assert "compute_ops[0] '': no input tensors" in message
    assert "compute_ops[0] '': ref 'MissingTensor' resolves to 'MissingTensor' which is not in tensors" in message


def test_validate_warns_on_unknown_operation_but_accepts_supported_structure():
    spec = MockOpSpec(
        op_spec_name="unknown_op_spec",
        dimensions={"i": 16},
        tensors={
            "Tensor0": _make_tensor("Tensor0", {"INPUT"}, [16]),
            "Tensor1": _make_tensor("Tensor1", {"OUTPUT"}, [16]),
        },
        compute_ops=[
            ComputeOp(
                op_func_name="mystery_op",
                ex_unit="sfp",
                input_tensor_names=["Tensor0"],
                output_tensor_names=["Tensor1"],
                num_cores=1,
            )
        ],
    )

    logger.info("Semantic validator case: expecting warning for unknown op '%s'", spec.compute_ops[0].op_func_name)

    with pytest.warns(UserWarning, match="unknown operation") as warning_record:
        MockOpSpecValidator().validate(spec)

    logger.warning(
        "Captured semantic warning for '%s': %s",
        spec.op_spec_name,
        warning_record[0].message,
    )
    logger.info("Unknown-op warning path completed for '%s'", spec.op_spec_name)


def test_validate_reports_invalid_execution_unit_and_incompatible_dtypes():
    spec = MockOpSpec(
        op_spec_name="bad_semantics_spec",
        dimensions={"i": 8},
        tensors={
            "Tensor0": _make_tensor("Tensor0", {"INPUT"}, [8]),
            "Tensor1": TensorDescriptor(
                name="Tensor1",
                roles={"INPUT"},
                shape=[8],
                stride=[1],
                dtype=torch.int32,
                device_size=[8],
                dim_map=[0],
                device_dtype="DataFormats.SEN169_INT32",
            ),
            "Tensor2": _make_tensor("Tensor2", {"OUTPUT"}, [8]),
        },
        compute_ops=[
            ComputeOp(
                op_func_name="add",
                ex_unit="vector_unit",
                input_tensor_names=["Tensor0", "Tensor1"],
                output_tensor_names=["Tensor2"],
                num_cores=1,
            )
        ],
    )

    logger.info("Semantic validator case: expecting invalid ex_unit and dtype mismatch for '%s'", spec.op_spec_name)

    with pytest.raises(ValueError) as exc_info:
        MockOpSpecValidator().validate(spec)

    message = str(exc_info.value)
    logger.error("Captured semantic validator error for '%s':\n%s", spec.op_spec_name, message)

    assert "invalid ex_unit 'vector_unit'" in message


def test_validate_reports_invalid_tensor_dtype():
    """Test that validator rejects tensors with invalid or unsupported dtypes."""
    spec = MockOpSpec(
        op_spec_name="invalid_dtype_spec",
        dimensions={"i": 16},
        tensors={
            "Tensor0": TensorDescriptor(
                name="Tensor0",
                roles={"INPUT"},
                shape=[16],
                stride=[1],
                dtype=torch.float64,  # Unsupported dtype
                device_size=[16],
                dim_map=[0],
                device_dtype="DataFormats.SEN169_FP64",
            ),
            "Tensor1": _make_tensor("Tensor1", {"OUTPUT"}, [16]),
        },
        compute_ops=[
            ComputeOp(
                op_func_name="add",
                ex_unit="sfp",
                input_tensor_names=["Tensor0"],
                output_tensor_names=["Tensor1"],
                num_cores=1,
            )
        ],
    )

    logger.info("Dtype validator case: expecting unsupported dtype error for '%s'", spec.op_spec_name)

    with pytest.raises(ValueError) as exc_info:
        MockOpSpecValidator().validate(spec)

    message = str(exc_info.value)
    logger.error("Captured dtype validator error for '%s':\n%s", spec.op_spec_name, message)

    assert "Tensor 'Tensor0': unsupported dtype" in message
    assert "float64" in message


def test_validate_reports_mixed_incompatible_dtypes_in_operation():
    """Test that validator rejects operations with incompatible input/output dtypes (e.g., float + int)."""
    spec = MockOpSpec(
        op_spec_name="mixed_dtype_spec",
        dimensions={"i": 32},
        tensors={
            "Tensor0": TensorDescriptor(
                name="Tensor0",
                roles={"INPUT"},
                shape=[32],
                stride=[1],
                dtype=torch.float32,  # float32 input
                device_size=[32],
                dim_map=[0],
                device_dtype="DataFormats.SEN169_FP32",
            ),
            "Tensor1": TensorDescriptor(
                name="Tensor1",
                roles={"INPUT"},
                shape=[32],
                stride=[1],
                dtype=torch.int32,  # int32 input - incompatible with float32
                device_size=[32],
                dim_map=[0],
                device_dtype="DataFormats.SEN169_INT32",
            ),
            "Tensor2": TensorDescriptor(
                name="Tensor2",
                roles={"OUTPUT"},
                shape=[32],
                stride=[1],
                dtype=torch.float16,  # float16 output - also incompatible
                device_size=[32],
                dim_map=[0],
                device_dtype="DataFormats.SEN169_FP16",
            ),
        },
        compute_ops=[
            ComputeOp(
                op_func_name="add",
                ex_unit="sfp",
                input_tensor_names=["Tensor0", "Tensor1"],
                output_tensor_names=["Tensor2"],
                num_cores=1,
            )
        ],
    )

    logger.info("Dtype consistency validator case: expecting incompatible dtype error for '%s'", spec.op_spec_name)

    with pytest.raises(ValueError) as exc_info:
        MockOpSpecValidator().validate(spec)

    message = str(exc_info.value)
    logger.error("Captured dtype consistency error for '%s':\n%s", spec.op_spec_name, message)

    assert "incompatible input/output dtypes" in message
    assert "float32" in message
    assert "int32" in message
    assert "float16" in message
