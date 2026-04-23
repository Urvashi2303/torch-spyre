"""
torch_spyre.mockdevice.backends.spyre.validator
-----------------------------------------------
MockOpSpecValidator - Validates MockOpSpec for Spyre device
"""
from __future__ import annotations
import warnings

import torch

from ...core import BaseGraphValidator, MockOpSpec
from ...logger import get_logger

logger = get_logger("MockOpSpecValidator")

KNOWN_OP_FUNCS = frozenset({
    "add",
    "mul",
    "gelu",
    "maxnonstick",
    "realdiv",
    "sumnonstick",
})

KNOWN_EX_UNITS = frozenset({"sfp", "pt"})
VALID_TORCH_DTYPES = frozenset({
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int32,
    torch.int64,
    torch.bool,
})

# Stick sizes for different dtypes (elements per 128-byte stick)
STICK_SIZES = {
    torch.float16: 64,    # 128 bytes / 2 bytes per element = 64
    torch.bfloat16: 64,   # 128 bytes / 2 bytes per element = 64
    torch.float32: 32,    # 128 bytes / 4 bytes per element = 32
    torch.int32: 32,      # 128 bytes / 4 bytes per element = 32
}

class MockOpSpecValidator(BaseGraphValidator):
    """
    Validates MockOpSpec for Spyre device.
    
    Extends BaseGraphValidator with Spyre-specific validation rules
    for operations, execution units, and device constraints.
    """

    def validate_device_specifics(self, spec: MockOpSpec, errors: list) -> None:
        logger.info("[FLOW] MockOpSpecValidator.validate_device_specifics() - Running Spyre-specific validation")

        self._check_tensor_dtypes(spec, errors)
        self._check_stick_alignment(spec, errors)
        self._check_compute_op_semantics(spec, errors)

    def _check_tensor_dtypes(self, spec: MockOpSpec, errors: list[str]) -> None:
        for name, td in spec.tensors.items():
            if not isinstance(td.dtype, torch.dtype):
                errors.append(
                    f"Tensor '{name}': dtype must be a valid torch.dtype, got {type(td.dtype).__name__}"
                )
                continue

            if td.dtype not in VALID_TORCH_DTYPES:
                errors.append(f"Tensor '{name}': unsupported dtype {td.dtype}")

    def _check_stick_alignment(self, spec: MockOpSpec, errors: list[str]) -> None:
        """
        Validate that tensor dimensions are aligned to stick boundaries.
        
        Spyre hardware stores data in 128-byte "sticks". The innermost dimension
        of each tensor must be divisible by the stick size for that dtype:
        - float16/bfloat16: 64 elements per stick (128 bytes / 2 bytes)
        - float32/int32: 32 elements per stick (128 bytes / 4 bytes)
        
        Misaligned dimensions cause DMA failures when the hardware attempts to
        transfer partial sticks.
        """
        for name, td in spec.tensors.items():
            # Skip dtypes without stick size requirements
            if td.dtype not in STICK_SIZES:
                continue
            
            stick_size = STICK_SIZES[td.dtype]
            
            # Check device_size (hardware view) - this is what DMA uses
            if td.device_size:
                innermost_dim = td.device_size[-1]
                if innermost_dim % stick_size != 0:
                    # Calculate suggested aligned size
                    aligned_size = ((innermost_dim // stick_size) + 1) * stick_size
                    errors.append(
                        f"Tensor '{name}': device innermost dimension {innermost_dim} "
                        f"not aligned to stick size {stick_size} for dtype {td.dtype}. "
                        f"This would cause failures on Spyre hardware. "
                        f"Consider padding to {aligned_size}."
                    )
                    logger.error(
                        f"[VALIDATION] Tensor '{name}' has misaligned dimension {innermost_dim}, "
                        f"expected multiple of {stick_size}"
                    )
            
            # Also check framework shape if device_size not available
            elif td.shape:
                innermost_dim = td.shape[-1]
                if innermost_dim % stick_size != 0:
                    aligned_size = ((innermost_dim // stick_size) + 1) * stick_size
                    errors.append(
                        f"Tensor '{name}': framework innermost dimension {innermost_dim} "
                        f"not aligned to stick size {stick_size} for dtype {td.dtype}. "
                        f"Consider padding to {aligned_size}."
                    )
                    logger.error(
                        f"[VALIDATION] Tensor '{name}' has misaligned shape {innermost_dim}, "
                        f"expected multiple of {stick_size}"
                    )

    def _check_compute_op_semantics(self, spec: MockOpSpec, errors: list[str]) -> None:
        for i, op in enumerate(spec.compute_ops):
            if op.op_func_name not in KNOWN_OP_FUNCS:
                warnings.warn(
                    f"compute_ops[{i}] '{op.op_func_name}': unknown operation; mock execution may be unsupported",
                    stacklevel=2,
                )
                logger.warning(
                    f"[FLOW] compute_ops[{i}] '{op.op_func_name}': unknown operation; "
                    "mock execution may be unsupported"
                )

            if not op.ex_unit or op.ex_unit not in KNOWN_EX_UNITS:
                errors.append(
                    f"compute_ops[{i}] '{op.op_func_name}': invalid ex_unit '{op.ex_unit}'. "
                    f"Expected one of {sorted(KNOWN_EX_UNITS)}"
                )

            # Validate all input/output tensors use the same dtype
            referenced_dtypes: list[torch.dtype] = []
            for ref in op.input_tensor_names + op.output_tensor_names:
                tname = ref.split("-")[0]
                td = spec.tensors.get(tname)
                if td is None or not isinstance(td.dtype, torch.dtype):
                    continue
                referenced_dtypes.append(td.dtype)

            if referenced_dtypes:
                first_dtype = referenced_dtypes[0]
                incompatible = [dtype for dtype in referenced_dtypes[1:] if dtype != first_dtype]
                if incompatible:
                    errors.append(
                        f"compute_ops[{i}] '{op.op_func_name}': incompatible input/output dtypes "
                        f"{[str(dtype) for dtype in referenced_dtypes]}"
                    )

