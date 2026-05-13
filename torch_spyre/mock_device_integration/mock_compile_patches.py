import os
import sympy
import torch
import torch._inductor.lowering as lowering

from contextlib import contextmanager
from torch._inductor.ir import Pointwise
from torch_spyre._inductor.views import compute_coordinates


MOCK_DEVICE_ENABLED = os.environ.get("TORCH_SPYRE_MOCK_DEVICE", "0") == "1"
MOCK_VERBOSE = os.environ.get("TORCH_SPYRE_MOCK_VERBOSE", "0") == "1"


def _mock_print(*args, **kwargs):
    if MOCK_DEVICE_ENABLED and MOCK_VERBOSE:
        print(*args, **kwargs)

_original_cat_lowering = None
_original_add_lowerings = {}
_original_mul_lowerings = {}
_original_full_lowering = None
_installed = False


def _mock_cat_lowering(inputs, dim=0):
    return lowering.make_fallback(torch.ops.aten.cat.default)(inputs, dim)


def _get_device_layout(inp):
    return getattr(getattr(inp, "layout", None), "device_layout", None)


def _get_stick_expr(inp):
    layout = _get_device_layout(inp)
    if layout is None:
        return None

    index = getattr(inp, "index", None)
    ranges = getattr(inp, "ranges", None)
    if index is None or ranges is None:
        return None

    device_coords = compute_coordinates(
        layout.device_size,
        layout.stride_map,
        ranges,
        index,
    )
    return device_coords[-1]


def _has_nonuniform_stick_indexing(inputs):
    stick_exprs = set()
    for inp in inputs:
        stick_expr = _get_stick_expr(inp)
        if stick_expr is not None and stick_expr != 0:
            stick_exprs.add(stick_expr)

    return len(stick_exprs) > 1


def _check_dim_order_compatibility(tensor_args):
    """
    Check if input tensors have incompatible dim_maps.
    In the actual hardware, operations fail when tensors have different
    dim_maps (physical memory layouts), even if they have the same logical shape.
    """
    if not MOCK_DEVICE_ENABLED:
        return
    
    # Get the dim_maps from the input tensor layouts
    dim_maps = []
    for arg in tensor_args:
        layout = _get_device_layout(arg)
        if layout is not None:
            dim_map = getattr(layout, "dim_map", None)
            if dim_map is not None and len(dim_map) > 0:
                dim_maps.append(tuple(dim_map))
    
    if len(dim_maps) < 2:
        return  # Need at least 2 tensors to check compatibility
    
    # Check if all dim_maps are identical
    # Different dim_maps mean incompatible physical layouts
    first_dim_map = dim_maps[0]
    for dim_map in dim_maps[1:]:
        if dim_map != first_dim_map:
            raise RuntimeError(
                "Spyre limitation: pointwise op with nonuniform stick indexing"
            )


def _wrap_add_lowering(original_lowering):
    def _mock_add_lowering(*args, **kwargs):
        tensor_args = [arg for arg in args if hasattr(arg, "layout")]

        # Check dim_order compatibility before lowering
        _check_dim_order_compatibility(tensor_args)

        result = original_lowering(*args, **kwargs)

        pointwise_args = list(tensor_args)
        if hasattr(result, "layout"):
            pointwise_args.append(result)

        if _has_nonuniform_stick_indexing(pointwise_args):
            raise RuntimeError(
                "Spyre limitation: pointwise op with nonuniform stick indexing"
            )

        return result

    return _mock_add_lowering


def install_mock_compile_patches():
    global _original_cat_lowering, _original_add_lowerings, _original_mul_lowerings, _original_full_lowering, _installed

    if not MOCK_DEVICE_ENABLED or _installed:
        return

    for aten_cat in lowering.get_overloads(torch.ops.aten.cat):
        _original_cat_lowering = lowering.lowerings.get(aten_cat)
        lowering.lowerings[aten_cat] = _mock_cat_lowering

    for aten_add in lowering.get_overloads(torch.ops.aten.add):
        original = lowering.lowerings.get(aten_add)
        if original is None:
            continue
        _original_add_lowerings[aten_add] = original
        lowering.lowerings[aten_add] = _wrap_add_lowering(original)

    for aten_mul in lowering.get_overloads(torch.ops.aten.mul):
        original = lowering.lowerings.get(aten_mul)
        if original is None:
            continue
        _original_mul_lowerings[aten_mul] = original
        lowering.lowerings[aten_mul] = _wrap_add_lowering(original)

    if hasattr(torch.ops.spyre, "full") and hasattr(torch.ops.spyre.full, "default"):
        _original_full_lowering = lowering.lowerings.get(torch.ops.spyre.full.default)
        lowering.lowerings[torch.ops.spyre.full.default] = lowering.make_fallback(torch.ops.aten.full.default)

    _installed = True
    _mock_print("[MOCK_DEVICE] Installed mock compile patches for aten.cat/add/mul overloads and spyre.full when available")

@contextmanager
def enable_mock_matmul_fallbacks():
    saved = {}
    fallback_mm = lowering.make_fallback(torch.ops.aten.mm.default)
    fallback_bmm = lowering.make_fallback(torch.ops.aten.bmm.default)

    def _mock_mm_lowering(*args, **kwargs):
        return fallback_mm(*args, **kwargs)

    def _mock_bmm_lowering(*args, **kwargs):
        return fallback_bmm(*args, **kwargs)

    for op, impl in (
        (torch.ops.aten.mm.default, _mock_mm_lowering),
        (torch.ops.aten.bmm.default, _mock_bmm_lowering),
    ):
        saved[op] = lowering.lowerings.get(op)
        lowering.lowerings[op] = impl
    try:
        yield
    finally:
        for op, impl in saved.items():
            if impl is None:
                lowering.lowerings.pop(op, None)
            else:
                lowering.lowerings[op] = impl


