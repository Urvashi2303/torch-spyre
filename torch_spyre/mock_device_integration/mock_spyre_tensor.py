# Copyright 2025 The Torch-Spyre Authors.
# Mock Spyre Tensor - Spoofs device as "spyre" while keeping data on CPU

"""
Mock Spyre Tensor Implementation

This creates a tensor that:
1. Reports device as "spyre" to PyTorch
2. Keeps actual data on CPU
3. Triggers Spyre inductor backend for compilation
4. Generates real SDSC JSON through the normal path

This is the key to making torch.compile() use the Spyre backend
without actual hardware.
"""

import torch
import os
import types

MOCK_DEVICE_ENABLED = os.environ.get('TORCH_SPYRE_MOCK_DEVICE', '0') == '1'
MOCK_VERBOSE = os.environ.get("TORCH_SPYRE_MOCK_VERBOSE", "0") == "1"


def _mock_print(*args, **kwargs):
    if MOCK_DEVICE_ENABLED and MOCK_VERBOSE:
        print(*args, **kwargs)

def _mock_has_debug_enabled():
    return MOCK_DEVICE_ENABLED and (
        MOCK_VERBOSE or os.environ.get("TORCH_SPYRE_MOCK_DEBUG_VIEW", "0") == "1"
    )

def _mock_debug_print(*args, **kwargs):
    if _mock_has_debug_enabled():
        print(*args, **kwargs)


class MockSpyreTensor(torch.Tensor):
    """
    A tensor subclass that reports device as 'spyre' but stores data on CPU.
    
    This tricks PyTorch's inductor into using the Spyre compilation backend,
    which generates real SDSC JSON through the normal code path.
    """
    
    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        # Create tensor on CPU
        if isinstance(data, torch.Tensor):
            cpu_tensor = data.cpu()
        else:
            cpu_tensor = torch.tensor(data, *args, **kwargs)
        
        # Create our subclass wrapping the CPU tensor
        return torch.Tensor._make_subclass(cls, cpu_tensor, require_grad=cpu_tensor.requires_grad)
    
    def __init__(self, data, *args, **kwargs):
        super().__init__()
        self._mock_device = torch.device("spyre", 0)
        if MOCK_DEVICE_ENABLED:
            try:
                cpu_data = data.cpu() if isinstance(data, torch.Tensor) else torch.tensor(data)
                _mock_print(
                    f"[MOCK_DEVICE] Created MockSpyreTensor: shape={tuple(cpu_data.shape)}, dtype={cpu_data.dtype}"
                )
            except Exception:
                _mock_print("[MOCK_DEVICE] Created MockSpyreTensor")
    
    @property
    def device(self):
        """Report device as 'spyre' to trick inductor"""
        # Handle case where _mock_device might not be set (e.g., after copy_())
        if not hasattr(self, '_mock_device'):
            self._mock_device = torch.device("spyre", 0)
        return self._mock_device

    @property
    def shape(self):
        logical_shape = getattr(self, "_mock_logical_shape", None)
        if logical_shape is not None:
            return torch.Size(logical_shape)
        return super().shape

    def size(self, *args):
        shape = self.shape
        if args:
            return shape[args[0]]
        return shape

    def stride(self, dim=None):
        logical_stride = getattr(self, "_mock_logical_stride", None)
        if logical_stride is None:
            return super().stride() if dim is None else super().stride(dim)
        if dim is None:
            return logical_stride
        return logical_stride[dim]

    def storage_offset(self):
        logical_offset = getattr(self, "_mock_storage_offset", None)
        if logical_offset is not None:
            return logical_offset
        return super().storage_offset()
    
    def device_tensor_layout(self):
        """
        Return a mock device tensor layout.
        This is required by Spyre inductor backend.
        """
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] device_tensor_layout() called for tensor shape={self.shape}")
        
        existing_layout = getattr(self, "_mock_device_layout", None)
        if existing_layout is not None:
            if MOCK_DEVICE_ENABLED:
                _mock_print("[MOCK_DEVICE] Returning preserved mock device layout")
            return existing_layout

        # Import here to avoid circular dependency
        try:
            from torch_spyre._C import SpyreTensorLayout
            
            # Use the 2-arg constructor which automatically calculates
            # stick-aware stride_map matching real Spyre behavior
            layout = SpyreTensorLayout(list(self.shape), self.dtype)
            
            if MOCK_DEVICE_ENABLED:
                _mock_print(f"[MOCK_DEVICE] Created SpyreTensorLayout: device_size={tuple(self.shape)}")
            return layout
        except Exception as e:
            if MOCK_DEVICE_ENABLED:
                _mock_print(f"[MOCK_DEVICE] Warning: Could not create SpyreTensorLayout: {e}")
                import traceback
                traceback.print_exc()
            # Return a mock object that has the basic attributes
            class MockLayout:
                def __init__(self):
                    self.device_size = tuple(self.shape) if hasattr(self, 'shape') else ()
                    self.dim_map = {}
                    self.stride_map = {}
            return MockLayout()
    
    def to(self, *args, **kwargs):
        """
        Override to() to handle device transfers.
        If target is 'spyre', return self (already "on" spyre).
        Otherwise, convert to regular tensor and transfer.
        """
        # Parse arguments
        device = None
        dtype = None
        for arg in args:
            if isinstance(arg, (str, torch.device)):
                device = torch.device(arg) if isinstance(arg, str) else arg
            elif isinstance(arg, torch.dtype):
                dtype = arg
        
        device = kwargs.get('device', device)
        dtype = kwargs.get('dtype', dtype)
        
        # If target is spyre, we're already there
        if device and 'spyre' in str(device):
            if MOCK_DEVICE_ENABLED:
                _mock_print(f"[MOCK_DEVICE] Tensor already on spyre device")
            return self
        
        # Otherwise, convert back to regular tensor and transfer
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] Converting MockSpyreTensor to regular tensor for device: {device}")
        regular_tensor = torch.Tensor._make_subclass(torch.Tensor, self)
        # Preserve dtype if not explicitly specified in the to() call
        if dtype is None and regular_tensor.dtype != self.dtype:
            regular_tensor = regular_tensor.to(dtype=self.dtype)
        return regular_tensor.to(*args, **kwargs)
    
    def cpu(self):
        """Return as regular CPU tensor, preserving dtype and logical view metadata"""
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] Converting MockSpyreTensor to CPU tensor")
        regular_tensor = torch.Tensor._make_subclass(torch.Tensor, self)
        logical_shape = getattr(self, "_mock_logical_shape", None)
        logical_stride = getattr(self, "_mock_logical_stride", None)
        logical_offset = getattr(self, "_mock_storage_offset", None)

        if (
            logical_shape is not None
            or logical_stride is not None
            or logical_offset is not None
        ):
            _mock_debug_print(
                "[MOCK_DEVICE] cpu() logical metadata "
                f"shape={logical_shape} stride={logical_stride} offset={logical_offset} "
                f"base_shape={tuple(regular_tensor.shape)} "
                f"base_stride={tuple(torch.Tensor.stride(regular_tensor))}"
            )

        if (
            logical_shape is not None
            or logical_stride is not None
            or logical_offset is not None
        ):
            base = regular_tensor
            if logical_offset is None:
                logical_offset = base.storage_offset()
            if logical_shape is None:
                logical_shape = tuple(base.shape)
            if logical_stride is None:
                logical_stride = tuple(base.stride())
            regular_tensor = torch.as_strided(
                base, size=logical_shape, stride=logical_stride, storage_offset=logical_offset
            )

        return regular_tensor
    
    def __repr__(self):
        return f"MockSpyreTensor({super().__repr__()}, device='{self.device}')"

    @classmethod
    def __torch_function__(cls, func, types_, args=(), kwargs=None):
        # Check stick dimension compatibility for pointwise operations
        func_name = getattr(func, "__name__", "")
        if func_name in {"add", "mul", "sub", "div"}:
            # Get all MockSpyreTensor inputs
            mock_tensors = [arg for arg in args if isinstance(arg, MockSpyreTensor)]
            
            if len(mock_tensors) >= 2:
                # Extract stick dimensions from each tensor's layout
                stick_dims = []
                for tensor in mock_tensors:
                    layout = getattr(tensor, "_mock_device_layout", None)
                    if layout is not None:
                        dim_map = getattr(layout, "dim_map", None)
                        if dim_map is not None and len(dim_map) > 0:
                            # Stick dimension is the last element of dim_map
                            stick_dim = dim_map[-1]
                            stick_dims.append(stick_dim)
                
                # Check if all stick dimensions are the same
                if len(stick_dims) >= 2:
                    first_stick_dim = stick_dims[0]
                    for stick_dim in stick_dims[1:]:
                        if stick_dim != first_stick_dim:
                            raise RuntimeError(
                                "Spyre limitation: pointwise op with nonuniform stick indexing"
                            )

        kwargs = kwargs or {}

        def _unwrap(x):
            if isinstance(x, MockSpyreTensor):
                return x.cpu()
            if isinstance(x, (tuple, list)):
                return type(x)(_unwrap(v) for v in x)
            if isinstance(x, dict):
                return {k: _unwrap(v) for k, v in x.items()}
            return x

        def _rewrap(x):
            if isinstance(x, MockSpyreTensor):
                return x
            if isinstance(x, torch.Tensor):
                try:
                    wrapped = MockSpyreTensor(x)
                except RuntimeError as e:
                    msg = str(e)
                    if (
                        "already associated to a python object of type FakeTensor" in msg
                        or "Creating a new Tensor subclass" in msg
                    ):
                        return x
                    raise
                wrapped._mock_logical_shape = tuple(x.shape)
                wrapped._mock_logical_stride = tuple(x.stride())
                wrapped._mock_storage_offset = x.storage_offset()
                return wrapped
            if isinstance(x, (tuple, list)):
                return type(x)(_rewrap(v) for v in x)
            if isinstance(x, dict):
                return {k: _rewrap(v) for k, v in x.items()}
            return x

        def _is_view_like(torch_func):
            name = getattr(torch_func, "__name__", "")
            return name in {
                "unsqueeze",
                "unsqueeze_",
                "squeeze",
                "squeeze_",
                "view",
                "reshape",
                "permute",
                "transpose",
                "t",
                "detach",
                "alias",
                "as_strided",
            }

        def _is_mock_only_cpu_boundary(torch_func, torch_args, torch_kwargs):
            name = getattr(torch_func, "__name__", "")
            has_mock_tensor_arg = any(
                isinstance(arg, MockSpyreTensor) for arg in torch_args
            )
            out_tensor = torch_kwargs.get("out") if isinstance(torch_kwargs, dict) else None
            has_mock_out = isinstance(out_tensor, MockSpyreTensor)

            if name in {"rsqrt", "__getitem__", "copy_", "cat", "sin", "cos", "arange"}:
                return has_mock_tensor_arg or has_mock_out

            if name == "addmm":
                return has_mock_out or has_mock_tensor_arg

            return False

        def _warn_and_run_cpu_fallback(torch_func, torch_args, torch_kwargs):
            from torch_spyre.ops.fallbacks import fallback_ops, warn_fallback

            cpu_args = _unwrap(torch_args)
            cpu_kwargs = _unwrap(torch_kwargs)

            fallback_name = None
            func_name = getattr(torch_func, "__name__", "")
            normalized_func_names = {
                func_name,
                func_name.replace("aten::", ""),
                func_name.replace("aten::", "").split(".")[0],
                func_name.split(".")[0],
            }
            for fallback_op in fallback_ops:
                op_name = getattr(fallback_op, "_name", "")
                if not isinstance(op_name, str):
                    continue
                normalized_op_names = {
                    op_name,
                    op_name.replace("aten::", ""),
                    op_name.replace("aten::", "").split(".")[0],
                    op_name.split(".")[0],
                    op_name.split(".")[-1],
                }
                if normalized_func_names & normalized_op_names:
                    fallback_name = op_name
                    break

            warn_fallback(fallback_name or func_name or str(torch_func))
            result = torch_func(*cpu_args, **cpu_kwargs)

            func_name = getattr(torch_func, "__name__", "")
            if func_name == "copy_" and torch_args:
                dst = torch_args[0]
                if isinstance(dst, MockSpyreTensor):
                    return dst

            if func_name in {"sin", "cos", "arange"}:
                out_tensor = torch_kwargs.get("out") if isinstance(torch_kwargs, dict) else None
                if isinstance(out_tensor, MockSpyreTensor):
                    return out_tensor

            return _rewrap(result)

        if getattr(func, "__name__", "") == "contiguous" and args and isinstance(args[0], MockSpyreTensor):
            return args[0]

        if _is_view_like(func):
            cpu_args = _unwrap(args)
            cpu_kwargs = _unwrap(kwargs)
            result = func(*cpu_args, **cpu_kwargs)
            return _rewrap(result)

        if _is_mock_only_cpu_boundary(func, args, kwargs):
            if MOCK_DEVICE_ENABLED:
                _mock_print(
                    f"[MOCK_DEVICE] mock-only CPU boundary fallback for {getattr(func, '__name__', func)}"
                )
            return _warn_and_run_cpu_fallback(func, args, kwargs)

        try:
            result = super().__torch_function__(func, types_, args, kwargs)
        except RuntimeError as e:
            msg = str(e)
            if "PyTorch is not linked with support for spyre devices" not in msg:
                raise
            if MOCK_DEVICE_ENABLED:
                _mock_print(
                    f"[MOCK_DEVICE] __torch_function__ fallback for {getattr(func, '__name__', func)}"
                )
            out_tensor = kwargs.get("out")
            cpu_kwargs = _unwrap(kwargs)
            cpu_out_tensor = cpu_kwargs.get("out") if isinstance(cpu_kwargs, dict) else None
            result = _warn_and_run_cpu_fallback(func, args, kwargs)

            if isinstance(out_tensor, MockSpyreTensor) and isinstance(cpu_out_tensor, torch.Tensor):
                return MockSpyreTensor(cpu_out_tensor)

            return result
        else:
            copy_src = getattr(result, "_mock_spyre_copy_src", None)
            if copy_src is not None:
                func_name = getattr(func, "__name__", "")
                if func_name == "copy_" and args:
                    dst = args[0]
                    dst_copy_src = getattr(dst, "_mock_spyre_copy_src", None)
                    if dst_copy_src is not None:
                        return _rewrap(_unwrap(dst_copy_src))
                    return _rewrap(_unwrap(dst))

                out_tensor = kwargs.get("out")
                if isinstance(out_tensor, MockSpyreTensor):
                    out_copy_src = getattr(out_tensor, "_mock_spyre_copy_src", None)
                    if out_copy_src is not None:
                        return _rewrap(_unwrap(out_copy_src))

                if func_name in {"add_", "mul_"} and args:
                    dst = args[0]
                    if isinstance(dst, MockSpyreTensor):
                        return dst

                return _rewrap(_unwrap(copy_src))

            if getattr(func, "__name__", "") in {"add_", "mul_"} and args:
                dst = args[0]
                if isinstance(dst, MockSpyreTensor):
                    return dst

            if any(isinstance(arg, MockSpyreTensor) for arg in args):
                func_name = getattr(func, "__name__", "")
                if func_name:
                    try:
                        from torch_spyre.ops.fallbacks import fallback_ops, warn_fallback

                        normalized_func_names = {
                            func_name,
                            func_name.replace("aten::", ""),
                            func_name.replace("aten::", "").split(".")[0],
                            func_name.split(".")[0],
                        }
                        for fallback_op in fallback_ops:
                            op_name = getattr(fallback_op, "_name", "")
                            if not isinstance(op_name, str):
                                continue
                            normalized_op_names = {
                                op_name,
                                op_name.replace("aten::", ""),
                                op_name.replace("aten::", "").split(".")[0],
                                op_name.split(".")[0],
                                op_name.split(".")[-1],
                            }
                            if normalized_func_names & normalized_op_names:
                                warn_fallback(op_name)
                                break
                    except Exception:
                        pass

            return result


def to_mock_spyre(tensor):
    """
    Convert a regular tensor to MockSpyreTensor.
    
    This is the key function that enables the mock device workflow:
    1. Takes CPU tensor
    2. Wraps in MockSpyreTensor (reports as spyre device)
    3. torch.compile() sees spyre device
    4. Uses Spyre inductor backend
    5. Generates real SDSC JSON
    """
    if isinstance(tensor, MockSpyreTensor):
        return tensor
    
    if MOCK_DEVICE_ENABLED:
        _mock_print(f"[MOCK_DEVICE] Converting tensor to mock Spyre device")
    
    return MockSpyreTensor(tensor)


# Monkey-patch tensor.to() to intercept "spyre" device requests
_original_tensor_to = torch.Tensor.to

def _patched_tensor_to(self, *args, **kwargs):
    """Patched version of tensor.to() that handles 'spyre' device"""
    # Check if target is spyre device - need to catch before PyTorch validates
    try:
        # Check if target is spyre device
        device = None
        for arg in args:
            if isinstance(arg, (str, torch.device)):
                device_str = str(arg)
                if 'spyre' in device_str.lower():
                    device = arg
                    break
        
        if not device:
            device = kwargs.get('device')
            if device and 'spyre' in str(device).lower():
                pass
            else:
                device = None
        
        # If target is spyre, convert to MockSpyreTensor
        if device:
            if MOCK_DEVICE_ENABLED:
                _mock_print(f"[MOCK_DEVICE] Intercepted .to('spyre') call")
            return to_mock_spyre(self)
        
        # Otherwise use original implementation
        return _original_tensor_to(self, *args, **kwargs)
    except RuntimeError as e:
        # If PyTorch rejects the device, check if it's spyre and handle it
        if 'spyre' in str(e).lower() or 'privateuse1' in str(e).lower():
            if MOCK_DEVICE_ENABLED:
                _mock_print(f"[MOCK_DEVICE] Caught PyTorch device error, converting to MockSpyreTensor")
            return to_mock_spyre(self)
        raise


def install_mock_device_patches():
    """Install patches to enable mock Spyre device"""
    if not MOCK_DEVICE_ENABLED:
        return
    
    _mock_print("[MOCK_DEVICE] Installing tensor.to() patch for Spyre device")
    torch.Tensor.to = _patched_tensor_to  # type: ignore
    _mock_print("[MOCK_DEVICE] Mock device patches installed successfully")


# Auto-install if mock device is enabled
if MOCK_DEVICE_ENABLED:
    install_mock_device_patches()
