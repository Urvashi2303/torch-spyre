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
