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
    """Mock cat lowering that bypasses spyre.overwrite decomposition."""
    # Use the standard inductor cat implementation which works on CPU
    # This avoids the spyre.overwrite operation that doesn't have a CPU kernel
    try:
        # Try to use the original PyTorch inductor cat implementation
        import torch._inductor.decomposition as decomp
        if hasattr(decomp, 'cat'):
            result = decomp.cat(inputs, dim)
            if result != NotImplemented:
                return result
    except Exception:
        pass
    
    # Fallback to aten.cat
    result = lowering.make_fallback(torch.ops.aten.cat.default)(inputs, dim)
    return result


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


def _patch_decompose_cat():
    """Disable decompose_cat for mock mode to force use of fallback lowering."""
    try:
        # Unregister the cat decomposition from the registry
        import torch._decomp as decomp_module
        from torch_spyre._inductor import decompositions
        
        # Try to remove cat from the decomposition registries
        for registry_name in ['decomposition_table', 'spyre_decompositions', 'spyre_decompositions_via_dispatchkey']:
            if hasattr(decompositions, registry_name):
                registry = getattr(decompositions, registry_name)
                if isinstance(registry, dict):
                    # Remove all cat-related entries
                    keys_to_remove = [k for k in registry.keys() if 'cat' in str(k).lower()]
                    for key in keys_to_remove:
                        del registry[key]
                        _mock_print(f"[MOCK_DEVICE] Removed {key} from {registry_name}")
        
        _mock_print("[MOCK_DEVICE] Unregistered cat decompositions")
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Failed to unregister cat decompositions: {e}")


def _patch_device_detection():
    """Patch device detection to return 'spyre' for MockSpyreTensors during compilation."""
    from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor
    
    try:
        import torch._inductor.ir as inductor_ir
        
        # Patch the decode_device function to handle MockSpyreTensor
        if hasattr(inductor_ir, 'decode_device'):
            original_decode_device = inductor_ir.decode_device
            
            def patched_decode_device(device):
                # If it's already a string starting with 'spyre', keep it
                if isinstance(device, str) and device.startswith('spyre'):
                    return device
                # Otherwise use original logic
                result = original_decode_device(device)
                # In mock mode, convert 'cpu' to 'spyre'
                if result == 'cpu' and MOCK_DEVICE_ENABLED:
                    return 'spyre'
                return result
            
            inductor_ir.decode_device = patched_decode_device
            _mock_print("[MOCK_DEVICE] Patched decode_device to return 'spyre' in mock mode")
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Warning: Could not patch decode_device: {e}")


def _patch_buffer_device():
    """Patch IR buffer/layout device to return 'spyre' in mock mode."""
    from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor
    
    try:
        import torch._inductor.ir as inductor_ir
        
        # Patch Buffer.get_device() to return 'spyre' if layout device is 'cpu' in mock mode
        if hasattr(inductor_ir, 'Buffer'):
            original_buffer_get_device = inductor_ir.Buffer.get_device
            
            def patched_buffer_get_device(self):
                device = original_buffer_get_device(self)
                # In mock mode, convert 'cpu' to 'spyre' for buffers
                if device == 'cpu' and MOCK_DEVICE_ENABLED:
                    # Check if this is part of a spyre compilation
                    # by checking if the layout has spyre-specific attributes
                    if hasattr(self, 'layout') and hasattr(self.layout, 'device'):
                        # Force device to be 'spyre' in mock mode
                        return 'spyre'
                return device
            
            inductor_ir.Buffer.get_device = patched_buffer_get_device
            _mock_print("[MOCK_DEVICE] Patched Buffer.get_device() to return 'spyre' in mock mode")
        
        # Also patch FixedLayout device property
        if hasattr(inductor_ir, 'FixedLayout'):
            FixedLayout = inductor_ir.FixedLayout
            # Store original device property
            original_device_property = FixedLayout.device
            
            def patched_device_getter(self):
                device = original_device_property.fget(self) if hasattr(original_device_property, 'fget') else self._device
                if device == 'cpu' and MOCK_DEVICE_ENABLED:
                    return 'spyre'
                return device
            
            def patched_device_setter(self, value):
                # In mock mode, always store as 'spyre' if trying to set 'cpu'
                if value == 'cpu' and MOCK_DEVICE_ENABLED:
                    value = 'spyre'
                if hasattr(original_device_property, 'fset'):
                    original_device_property.fset(self, value)
                else:
                    self._device = value
            
            FixedLayout.device = property(patched_device_getter, patched_device_setter)
            _mock_print("[MOCK_DEVICE] Patched FixedLayout.device property for mock mode")
            
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Warning: Could not patch buffer device: {e}")


def install_mock_compile_patches():
    global _original_cat_lowering, _original_add_lowerings, _original_mul_lowerings, _original_full_lowering, _installed

    if not MOCK_DEVICE_ENABLED or _installed:
        return
    
    # Patch device detection
    _patch_device_detection()
    _patch_buffer_device()
    
    # Patch decompose_cat to return NotImplemented
    _patch_decompose_cat()

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

    # Don't register fallback for spyre.full in mock mode - it conflicts with decomposition
    # if hasattr(torch.ops.spyre, "full") and hasattr(torch.ops.spyre.full, "default"):
    #     _original_full_lowering = lowering.lowerings.get(torch.ops.spyre.full.default)
    #     lowering.lowerings[torch.ops.spyre.full.default] = lowering.make_fallback(torch.ops.aten.full.default)

    # Patch torch.compile to wrap results in MockSpyreTensor
    _install_compile_result_wrapper()

    _installed = True
    _mock_print("[MOCK_DEVICE] Installed mock compile patches for aten.cat/add/mul overloads and spyre.full when available")


def _install_compile_result_wrapper():
    """Wrap torch.compile results to return MockSpyreTensors when inputs are MockSpyreTensors."""
    from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor
    import functools
    
    original_compile = torch.compile
    
    @functools.wraps(original_compile)
    def wrapped_compile(model=None, *,  fullgraph=False, dynamic=None, backend="inductor", mode=None, options=None, disable=False):
        compiled_fn = original_compile(model, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode, options=options, disable=disable)
        
        @functools.wraps(compiled_fn)
        def wrapper(*args, **kwargs):
            # Check if any input is a MockSpyreTensor
            has_mock_input = any(
                isinstance(arg, MockSpyreTensor) for arg in args
            ) or any(
                isinstance(v, MockSpyreTensor) for v in kwargs.values()
            )
            
            # Call the compiled function
            result = compiled_fn(*args, **kwargs)
            
            # If inputs were MockSpyreTensors, wrap tensor results
            if has_mock_input:
                def wrap_result(x):
                    if isinstance(x, torch.Tensor) and not isinstance(x, MockSpyreTensor):
                        if MOCK_VERBOSE:
                            _mock_print(f"[MOCK_DEVICE] Wrapping compile result in MockSpyreTensor")
                        wrapped = MockSpyreTensor.__new__(MockSpyreTensor, x)
                        wrapped._mock_device = torch.device("spyre", 0)
                        return wrapped
                    elif isinstance(x, (tuple, list)):
                        return type(x)(wrap_result(item) for item in x)
                    return x
                
                return wrap_result(result)
            
            return result
        
        return wrapper
    
    torch.compile = wrapped_compile
    _mock_print("[MOCK_DEVICE] Installed torch.compile result wrapper")

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


