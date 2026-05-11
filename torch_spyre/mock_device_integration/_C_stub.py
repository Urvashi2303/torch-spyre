# Copyright 2025 The Torch-Spyre Authors.
import functools
# Stub implementation of _C module for when C++ extensions are not available

"""
Stub implementation of torch_spyre._C module.
This provides minimal dummy implementations to allow imports when
the wheel is built with stub dependencies.

MVP Mock Device: This stub now includes basic device registration
to enable SDSC JSON generation without actual hardware.
"""

import warnings
import os
import types
import math
import sympy

# Check if mock device is enabled
MOCK_DEVICE_ENABLED = os.environ.get('TORCH_SPYRE_MOCK_DEVICE', '0') == '1'
MOCK_VERBOSE = os.environ.get("TORCH_SPYRE_MOCK_VERBOSE", "0") == "1"


def _mock_print(*args, **kwargs):
    if MOCK_DEVICE_ENABLED and MOCK_VERBOSE:
        print(*args, **kwargs)


# Show warning once when this stub is imported
if not MOCK_DEVICE_ENABLED:
    warnings.warn(
        "Using stub implementation of torch_spyre._C module. "
        "Full functionality requires actual IBM Spyre libraries. "
        "Set TORCH_SPYRE_MOCK_DEVICE=1 to enable mock device.",
        RuntimeWarning,
        stacklevel=2
    )
else:
    _mock_print("[MOCK_DEVICE] Spyre mock device enabled - SDSC JSON generation active")


class DataFormats:
    """Stub DataFormats enum-like class with elems_per_stick() method"""
    
    def __init__(self, value, name, elems=1):
        self._value = value
        self._name = name
        self._elems = elems
    
    def __int__(self):
        return self._value
    
    def __repr__(self):
        return f"DataFormats.{self._name}"
    
    def __eq__(self, other):
        if isinstance(other, DataFormats):
            return self._value == other._value
        return self._value == other
    
    def __hash__(self):
        return hash(self._value)
    
    @property
    def value(self):
        return self._value
    
    @property
    def name(self):
        return self._name
    
    def elems_per_stick(self):
        """Return number of elements per stick for this data format"""
        return self._elems

# Create enum-like instances
DataFormats.SEN169_FP16 = DataFormats(0, "SEN169_FP16", elems=64)
DataFormats.IEEE_FP32 = DataFormats(1, "IEEE_FP32", elems=32)
DataFormats.SEN169_FP8 = DataFormats(2, "SEN169_FP8", elems=64)


class SpyreTensorLayout:
    """Stub SpyreTensorLayout class with required attributes for inductor"""
    def __init__(self, *args, **kwargs):
        # Handle 3 calling patterns based on number of args
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] SpyreTensorLayout.__init__ called with {len(args)} args: {args}")
        if len(args) == 2:
            # Pattern 1: SpyreTensorLayout(size, dtype)
            host_size = list(args[0]) if hasattr(args[0], '__iter__') else [args[0]]

            device_dtype = get_device_dtype(args[1])
            self.device_dtype = device_dtype
            self.data_format = device_dtype

            stick_size = device_dtype.elems_per_stick()

            if len(host_size) == 0:
                self.device_size = [1, stick_size]
                self.dim_map = [-1, -1]
                self.stride_map = [-1, -1]
            elif len(host_size) == 1:
                self.device_size = [math.ceil(host_size[0] / stick_size), stick_size]
                self.dim_map = [0, 0]
                self.stride_map = [stick_size, 1]
            else:
                host_stride = []
                stride = 1
                for i in range(len(host_size) - 1, -1, -1):
                    host_stride.insert(0, stride)
                    stride *= host_size[i]

                stick_dim = len(host_size) - 1
                outer = math.ceil(host_size[stick_dim] / stick_size)
                prefix_dims = [host_size[i] for i in range(len(host_size) - 1)]

                if len(host_size) == 2:
                    self.device_size = [outer] + prefix_dims + [stick_size]
                    self.dim_map = [stick_dim, 0, stick_dim]
                    self.stride_map = [stick_size, host_stride[0], 1]
                else:
                    leading_dims = host_size[:-1]
                    self.device_size = [leading_dims[-1], outer] + list(reversed(leading_dims[:-1])) + [stick_size]
                    self.dim_map = [len(host_size) - 2, stick_dim] + list(range(len(host_size) - 3, -1, -1)) + [stick_dim]
                    self.stride_map = [host_stride[len(host_size) - 2], stick_size, *[host_stride[idx] for idx in range(len(host_size) - 3, -1, -1)], 1]
            
        elif len(args) == 4:
            # Two possible patterns - detect by checking if arg[3] is already a DataFormats
            if isinstance(args[3], DataFormats):
                # Pattern B: SpyreTensorLayout(device_size, dim_map, stride_map, device_dtype)
                if MOCK_DEVICE_ENABLED:
                    _mock_print(f"[MOCK_DEVICE] Pattern B: device_size={args[0]}, dim_map={args[1]}, stride_map={args[2]}, device_dtype={args[3]}")
                self.device_size = tuple(args[0]) if hasattr(args[0], '__iter__') else (args[0],)
                # dim_map should always be a list (matches C++ std::vector<int32_t>)
                self.dim_map = list(args[1]) if hasattr(args[1], '__iter__') else []
                # stride_map should always be a list (matches C++ std::vector<int32_t>)
                self.stride_map = list(args[2]) if hasattr(args[2], '__iter__') and not isinstance(args[2], str) else []
                self.device_dtype = args[3]
                self.data_format = args[3]
            else:
                # Pattern A: SpyreTensorLayout(size, stride, dtype, dim_order)
                # This matches C++ init(host_size, host_strides, dtype, dim_order) at line 141
                if MOCK_DEVICE_ENABLED:
                    _mock_print(f"[MOCK_DEVICE] Pattern A: size={args[0]}, stride={args[1]}, dtype={args[2]}, dim_order={args[3]}")
                
                host_size = list(args[0]) if hasattr(args[0], '__iter__') else [args[0]]
                host_stride = list(args[1]) if hasattr(args[1], '__iter__') else [args[1]]
                dim_order = args[3] if isinstance(args[3], list) else list(range(len(host_size)))
                
                # Convert torch dtype to device dtype
                device_dtype = get_device_dtype(args[2])
                self.device_dtype = device_dtype
                self.data_format = device_dtype
                
                stick_size = device_dtype.elems_per_stick()
                sparse = dim_order[-1] == -1

                if sparse:
                    logical_order = list(dim_order[:-1])
                    if len(host_size) == 0 or len(logical_order) == 0:
                        self.device_size = [1, stick_size]
                        self.dim_map = [-1, -1]
                        self.stride_map = [-1, 1]
                    else:
                        self.device_size = [host_size[logical_order[-1]], 1] + [host_size[d] for d in logical_order[:-1]] + [stick_size]
                        self.dim_map = [logical_order[-1], -1] + logical_order[:-1] + [-1]
                        self.stride_map = [host_stride[logical_order[-1]], 1] + [host_stride[d] for d in logical_order[:-1]] + [1]
                else:
                    stick_dim = dim_order[-1]
                    outer = math.ceil(host_size[stick_dim] / stick_size)
                    nonstick_dims = list(dim_order[:-1])

                    if len(host_size) == 1:
                        self.device_size = [outer, stick_size]
                        self.dim_map = [stick_dim, stick_dim]
                        self.stride_map = [stick_size, 1]
                    elif len(host_size) == 2:
                        self.device_size = [outer, host_size[nonstick_dims[0]], stick_size]
                        self.dim_map = [stick_dim, nonstick_dims[0], stick_dim]
                        self.stride_map = [stick_size, host_stride[nonstick_dims[0]], 1]
                    else:
                        self.device_size = [outer, host_size[nonstick_dims[1]], host_size[nonstick_dims[0]], stick_size]
                        self.dim_map = [nonstick_dims[1], stick_dim, nonstick_dims[0], stick_dim]
                        self.stride_map = [stick_size * host_size[stick_dim], host_stride[nonstick_dims[1]], host_stride[nonstick_dims[0]], 1]

                if MOCK_DEVICE_ENABLED:
                    _mock_print(f"[MOCK_DEVICE] Pattern A computed: dim_map={self.dim_map}, device_size={self.device_size}, stride_map={self.stride_map}")
            
        else:
            # Pattern 3: Named parameters (fallback)
            self.device_size = kwargs.get('device_size', ())
            self.dim_map = kwargs.get('dim_map', {})
            self.stride_map = kwargs.get('stride_map', [])
            data_format = kwargs.get('data_format', DataFormats.IEEE_FP32)
            self.device_dtype = data_format
            self.data_format = data_format
        
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] SpyreTensorLayout created: device_size={self.device_size}")
    
    def _dim_map_to_stride_map(self, dim_map, host_size, host_stride, device_size):
        """
        Python port of C++ dim_map_to_stride_map from spyre_tensor_impl.cpp lines 111-131.
        
        Args:
            dim_map: List of dimension mappings from device to host
            host_size: Host tensor dimensions
            host_stride: Host tensor strides
            device_size: Device tensor dimensions (with stick tiling)
        
        Returns:
            List of stride values (INCLUDING -1 values, matching C++ behavior)
        """
        if len(dim_map) == 0:
            return []
        
        n = len(dim_map)
        stride_map = [-1] * n
        last_stride = [-1] * n
        
        for j in range(n - 1, -1, -1):
            d = dim_map[j]
            if d == -1 or host_size[d] == 1:
                stride_map[j] = -1
            elif last_stride[d] == -1:
                stride_map[j] = host_stride[d]
                last_stride[d] = stride_map[j] * device_size[j]
            else:
                stride_map[j] = last_stride[d]
                last_stride[d] = stride_map[j] * device_size[j]
        
        # Return the full stride_map including -1 values (matches C++ line 130)
        return stride_map
    
    def elems_per_stick(self):
        return self.device_dtype.elems_per_stick()

    def host_stick_dim(self):
        if not self.dim_map:
            return None
        if self.dim_map[-1] == -1:
            return None
        return self.dim_map[-1]

    def __eq__(self, other):
        return (
            isinstance(other, SpyreTensorLayout)
            and list(self.device_size) == list(other.device_size)
            and list(self.dim_map) == list(other.dim_map)
            and list(self.stride_map) == list(other.stride_map)
            and self.device_dtype == other.device_dtype
        )

    def __str__(self):
        return (
            f"SpyreTensorLayout(device_size={list(self.device_size)}, "
            f"dim_map ={list(self.dim_map)}, "
            f"stride_map ={list(self.stride_map)}, "
            f"device_dtype={self.device_dtype})"
        )

    def __repr__(self):
        return self.__str__()


def get_spyre_tensor_layout(tensor):
    """Return a mock SpyreTensorLayout for mock spyre tensors."""
    layout = getattr(tensor, "_mock_device_layout", None)
    if layout is not None:
        return layout
    try:
        return SpyreTensorLayout(list(tensor.shape), tensor.dtype)
    except Exception:
        return None


def to_with_layout(tensor, layout):
    """Create a mock spyre tensor and preserve the requested layout."""
    from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor

    result = MockSpyreTensor(tensor)
    try:
        object.__setattr__(result, "_mock_device_layout", layout)
    except Exception:
        pass
    return result


def empty_with_layout(*args, **kwargs):
    """Create a mock spyre tensor and preserve the requested layout."""
    import torch
    from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor

    size = args[0] if args else kwargs.get("size")
    layout = args[1] if len(args) > 1 else kwargs.get("device_layout")
    dtype = args[2] if len(args) > 2 else kwargs.get("dtype", torch.float32)

    result = MockSpyreTensor(torch.empty(size, dtype=dtype))
    try:
        object.__setattr__(result, "_mock_device_layout", layout)
    except Exception:
        pass
    return result


def spyre_empty_with_layout(*args, **kwargs):
    """
    Stub for spyre_empty_with_layout.
    Real signature: spyre_empty_with_layout(size, stride, dtype, layout)
    """
    import torch
    if len(args) >= 3:
        # args[0] = size, args[1] = stride, args[2] = dtype, args[3] = layout
        size, stride, dtype = args[0], args[1], args[2]
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] spyre_empty_with_layout: size={size}, dtype={dtype}")
        # Create MockSpyreTensor with correct dtype
        from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor
        return MockSpyreTensor(torch.empty(size, dtype=dtype))
    else:
        # Fallback for unexpected call pattern
        return torch.empty(*args[:1])


def reinterpret_tensor(tensor, *args, **kwargs):
    """Stub function"""
    return tensor


def reinterpret_tensor_with_layout(tensor, *args, **kwargs):
    """Stub function"""
    return tensor


def encode_constant(value, format):
    """Stub function"""
    return value


def convert_artifacts(*args, **kwargs):
    """Stub function"""
    pass


def launch_kernel(*args, **kwargs):
    """Stub function"""
    raise NotImplementedError(
        "launch_kernel requires actual torch_spyre._C module. "
        "Rebuild with IBM Spyre libraries for full functionality."
    )


def _patch_torch_accelerator_for_mock():
    """Patch torch.accelerator APIs that otherwise call into missing C++ spyre support."""
    if not MOCK_DEVICE_ENABLED:
        return

    try:
        import torch
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Warning: could not import torch for accelerator patching: {e}")
        return

    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        return

    if getattr(accelerator, "_spyre_mock_patched", False):
        return

    def _mock_current_device_index():
        return 0

    def _mock_current_stream(device=None):
        try:
            import torch_spyre
            if hasattr(torch_spyre, "current_stream"):
                return torch_spyre.current_stream(device)
        except Exception:
            pass
        return types.SimpleNamespace(device=lambda: torch.device("spyre", 0))

    def _mock_set_device_index(device=None):
        return None

    def _mock_synchronize(device=None):
        return None

    accelerator.current_device_index = _mock_current_device_index
    accelerator.current_stream = _mock_current_stream
    accelerator.set_device_index = _mock_set_device_index
    accelerator.synchronize = _mock_synchronize
    accelerator._spyre_mock_patched = True
    _mock_print("[MOCK_DEVICE] Patched torch.accelerator APIs for mock spyre backend")


def _patch_fake_tensor_ops_for_mock():
    """Patch selected fake-tensor execution paths to use CPU semantics in mock Spyre mode."""
    if not MOCK_DEVICE_ENABLED:
        return

    try:
        import operator
        from torch._dynamo import utils as dynamo_utils
        import torch
        import torch._subclasses.fake_tensor as fake_tensor_mod
        from torch._subclasses.fake_tensor import FakeTensor
        from torch._subclasses.functional_tensor import FunctionalTensor
        from torch.fx import Interpreter
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Warning: could not import fake-tensor utilities for patching: {e}")
        return

    if getattr(dynamo_utils, "_spyre_mock_patched", False):
        return

    try:
        full_default = torch.ops.spyre.full.default
    except Exception:
        full_default = None
    else:
        if not getattr(full_default, "_spyre_mock_cpu_wrapped", False):
            original_full_default = full_default

            @functools.wraps(original_full_default)
            def _spyre_full_default_with_mock_cpu_fallback(*args, **kwargs):
                device = None
                if len(args) > 2:
                    device = args[2]
                elif isinstance(kwargs, dict):
                    device = kwargs.get("device")

                if getattr(device, "type", None) == "cpu":
                    size = args[0] if len(args) > 0 else kwargs.get("size")
                    fill_value = args[1] if len(args) > 1 else kwargs.get("fill_value")
                    dtype = args[3] if len(args) > 3 else kwargs.get("dtype")
                    if MOCK_DEVICE_ENABLED:
                        _mock_print("[MOCK_DEVICE] direct CPU fallback for torch.ops.spyre.full.default(..., device=cpu)")
                    return torch.full(size, fill_value, dtype=dtype, device="cpu")

                return original_full_default(*args, **kwargs)

            setattr(_spyre_full_default_with_mock_cpu_fallback, "_spyre_mock_cpu_wrapped", True)
            torch.ops.spyre.full.default = _spyre_full_default_with_mock_cpu_fallback

    original_run_node = dynamo_utils.run_node
    original_find_common_device = FakeTensor._find_common_device
    original_interpreter_call_function = Interpreter.call_function
    original_fake_tensor_torch_function = getattr(FakeTensor, "__torch_function__", None)
    try:
        import torch._functorch._aot_autograd.functional_utils as aot_functional_utils
        import torch._functorch._aot_autograd.runtime_wrappers as aot_runtime_wrappers
    except Exception:
        aot_functional_utils = None
        aot_runtime_wrappers = None
        original_gen_alias_from_base = None
    else:
        original_gen_alias_from_base = aot_functional_utils.gen_alias_from_base

    def _is_mock_spyre_tensor(obj):
        return isinstance(obj, FakeTensor) and getattr(getattr(obj, "device", None), "type", None) == "spyre"

    def _logical_shape(obj):
        shape = getattr(obj, "_mock_logical_shape", None)
        if shape is not None:
            return tuple(shape)
        try:
            return tuple(obj.shape)
        except Exception:
            return None

    def _contains_mock_spyre_tensor(obj):
        if _is_mock_spyre_tensor(obj):
            return True
        if isinstance(obj, FunctionalTensor):
            try:
                inner = obj.from_functional()
            except Exception:
                inner = None
            return _contains_mock_spyre_tensor(inner)
        if isinstance(obj, (tuple, list)):
            return any(_contains_mock_spyre_tensor(item) for item in obj)
        if isinstance(obj, dict):
            return any(_contains_mock_spyre_tensor(item) for item in obj.values())
        return False

    def _to_cpu_tree(obj):
        if _is_mock_spyre_tensor(obj):
            return obj.to(device="cpu")
        if isinstance(obj, FunctionalTensor):
            inner = obj.from_functional()
            return _to_cpu_tree(inner)
        if isinstance(obj, tuple):
            return tuple(_to_cpu_tree(item) for item in obj)
        if isinstance(obj, list):
            return [_to_cpu_tree(item) for item in obj]
        if isinstance(obj, dict):
            return {key: _to_cpu_tree(value) for key, value in obj.items()}
        return obj

    def _describe_mock_result(obj):
        if isinstance(obj, torch.Tensor):
            return (
                f"type={type(obj).__name__} "
                f"device={getattr(getattr(obj, 'device', None), 'type', None)} "
                f"shape={tuple(obj.shape)} "
                f"logical_shape={getattr(obj, '_mock_logical_shape', None)} "
                f"logical_stride={getattr(obj, '_mock_logical_stride', None)} "
                f"logical_offset={getattr(obj, '_mock_storage_offset', None)}"
            )
        if isinstance(obj, (tuple, list)):
            return type(obj).__name__ + "[" + ", ".join(_describe_mock_result(v) for v in obj) + "]"
        if isinstance(obj, dict):
            return "dict{" + ", ".join(f"{k}: {_describe_mock_result(v)}" for k, v in obj.items()) + "}"
        return repr(obj)

    def _to_spyre_tree(obj):
        if isinstance(obj, tuple):
            return tuple(_to_spyre_tree(item) for item in obj)
        if isinstance(obj, list):
            return [_to_spyre_tree(item) for item in obj]
        if isinstance(obj, dict):
            return {key: _to_spyre_tree(value) for key, value in obj.items()}
        if isinstance(obj, torch.Tensor) and getattr(getattr(obj, "device", None), "type", None) == "cpu":
            if type(obj).__name__ == "FakeTensor":
                if os.environ.get("TORCH_SPYRE_MOCK_DEBUG_VIEW", "0") == "1":
                    _mock_print("[MOCK_DEVICE] _to_spyre_tree leaving FakeTensor on cpu to avoid invalid subclass rewrap")
                return obj
            from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor
            wrapped = MockSpyreTensor(obj)
            if os.environ.get("TORCH_SPYRE_MOCK_DEBUG_VIEW", "0") == "1":
                _mock_print(f"[MOCK_DEVICE] _to_spyre_tree wrapped {_describe_mock_result(wrapped)}")
            return wrapped
        return obj

    def _run_node_with_cpu_fallback(tracer, node, args, kwargs, nnmodule):
        target = getattr(node, "target", None)
        if node.op == "call_method" and target == "copy_" and (
            _contains_mock_spyre_tensor(args) or _contains_mock_spyre_tensor(kwargs)
        ):
            if MOCK_DEVICE_ENABLED:
                _mock_print("[MOCK_DEVICE] Dynamo fake-tensor direct fallback for method copy_ on spyre")
            dst = args[0]
            src = args[1] if len(args) > 1 else None
            if _is_mock_spyre_tensor(dst) and src is not None:
                try:
                    object.__setattr__(dst, "_mock_spyre_copy_src", src)
                except Exception:
                    pass
            return dst

        if node.op == "call_method" and target == "matmul" and (
            _contains_mock_spyre_tensor(args) or _contains_mock_spyre_tensor(kwargs)
        ):
            if MOCK_DEVICE_ENABLED:
                _mock_print("[MOCK_DEVICE] Dynamo fake-tensor direct CPU fallback for method matmul on spyre")
            lhs = args[0]
            rhs = args[1] if len(args) > 1 else None
            cpu_kwargs = _to_cpu_tree(kwargs)

            def _maybe_mock_shape(x):
                return _logical_shape(x)

            if lhs is not None and rhs is not None:
                lhs_shape = _maybe_mock_shape(lhs)
                rhs_shape = _maybe_mock_shape(rhs)

                if lhs_shape is not None and rhs_shape is not None:
                    if (
                        len(lhs_shape) >= 2
                        and len(rhs_shape) >= 2
                        and lhs_shape[-1] != rhs_shape[-2]
                        and lhs_shape[-1] == rhs_shape[-1]
                    ):
                        if MOCK_DEVICE_ENABLED:
                            _mock_print(
                                "[MOCK_DEVICE] Dynamo matmul fake fallback: fixing stale RHS transpose shape before CPU matmul"
                            )
                        rhs = rhs.transpose(-1, -2)

            if hasattr(lhs, "cpu") and rhs is not None and hasattr(rhs, "cpu"):
                result = lhs.cpu().matmul(rhs.cpu(), **cpu_kwargs)
                return _to_spyre_tree(result)

        def _cpu_fallback_from_run_node():
            cpu_args = _to_cpu_tree(args)
            cpu_kwargs = _to_cpu_tree(kwargs)

            if target is operator.getitem:
                if MOCK_DEVICE_ENABLED:
                    _mock_print("[MOCK_DEVICE] Dynamo fake-tensor CPU fallback for builtins.getitem on spyre")
                result = target(*cpu_args, **cpu_kwargs)
                return _to_spyre_tree(result)

            if node.op == "call_method" and isinstance(target, str):
                if target == "copy_":
                    if MOCK_DEVICE_ENABLED:
                        _mock_print("[MOCK_DEVICE] Dynamo fake-tensor CPU fallback for method copy_ on spyre")
                    dst = args[0]
                    src = args[1] if len(args) > 1 else None
                    if _is_mock_spyre_tensor(dst) and src is not None:
                        try:
                            object.__setattr__(dst, "_mock_spyre_copy_src", src)
                        except Exception:
                            pass
                    return dst
                if MOCK_DEVICE_ENABLED:
                    _mock_print(f"[MOCK_DEVICE] Dynamo fake-tensor CPU fallback for method {target} on spyre")
                result = getattr(cpu_args[0], target)(*cpu_args[1:], **cpu_kwargs)
                return _to_spyre_tree(result)

            if callable(target):
                target_name = getattr(target, "__name__", "")
                target_overloadpacket = getattr(target, "overloadpacket", None)
                target_packet_name = getattr(target_overloadpacket, "__name__", "")

                if target_packet_name == "full":
                    if MOCK_DEVICE_ENABLED:
                        _mock_print("[MOCK_DEVICE] Dynamo fake-tensor CPU fallback for spyre.full on spyre")
                    size = cpu_args[0]
                    fill_value = cpu_args[1]
                    device = cpu_args[2]
                    dtype = cpu_args[3] if len(cpu_args) > 3 else cpu_kwargs.get("dtype")
                    return _to_spyre_tree(torch.full(size, fill_value, dtype=dtype, device=device))

                if MOCK_DEVICE_ENABLED:
                    _mock_print(
                        f"[MOCK_DEVICE] Dynamo fake-tensor CPU fallback for "
                        f"{target_name or repr(target)} on spyre"
                    )
                if target_name == "addmm":
                    out_arg = kwargs.get("out")
                    if out_arg is not None and _is_mock_spyre_tensor(out_arg):
                        cpu_input = cpu_args[0] if len(cpu_args) > 0 else None
                        cpu_mat1 = cpu_args[1] if len(cpu_args) > 1 else None
                        cpu_mat2 = cpu_args[2] if len(cpu_args) > 2 else None
                        alpha = kwargs.get("alpha", 1)
                        beta = kwargs.get("beta", 1)
                        computed = torch.addmm(cpu_input, cpu_mat1, cpu_mat2, beta=beta, alpha=alpha)
                        try:
                            object.__setattr__(out_arg, "_mock_spyre_copy_src", computed)
                        except Exception:
                            pass
                        return computed

                result = target(*cpu_args, **cpu_kwargs)
                if target_name == "addmm" and len(args) >= 1 and _contains_mock_spyre_tensor(args[0]):
                    return _to_spyre_tree(result)
                return _to_spyre_tree(result)

            raise

        try:
            result = original_run_node(tracer, node, args, kwargs, nnmodule)
            if os.environ.get("TORCH_SPYRE_MOCK_DEBUG_VIEW", "0") == "1" and (
                _contains_mock_spyre_tensor(args) or _contains_mock_spyre_tensor(kwargs)
            ):
                _mock_print(f"[MOCK_DEVICE] run_node result {_describe_mock_result(result)}")
            return result
        except RuntimeError as e:
            msg = str(e)
            if "PyTorch is not linked with support for spyre devices" not in msg and (
                "does not have a kernel registered for cpu" not in msg or "full" not in msg
            ):
                raise
            if not _contains_mock_spyre_tensor(args) and not _contains_mock_spyre_tensor(kwargs):
                raise
            return _cpu_fallback_from_run_node()

    def _find_common_device_with_mock_spyre_support(func, flat_args):
        try:
            return original_find_common_device(func, flat_args)
        except RuntimeError as e:
            msg = str(e)
            if "Unhandled FakeTensor Device Propagation" not in msg:
                raise

            devices = []
            has_spyre = False
            has_cpu = False
            for arg in flat_args:
                if isinstance(arg, FakeTensor):
                    device = getattr(arg, "device", None)
                    if device is not None:
                        devices.append(device)
                        if device.type == "spyre":
                            has_spyre = True
                        elif device.type == "cpu":
                            has_cpu = True

            if has_spyre and has_cpu:
                if MOCK_DEVICE_ENABLED:
                    _mock_print("[MOCK_DEVICE] FakeTensor device propagation fallback: preferring spyre over cpu")
                return fake_tensor_mod.torch.device("spyre", 0), False

            raise

    def _interpreter_call_function_with_mock_spyre_fallback(self, target, args, kwargs):
        try:
            result = original_interpreter_call_function(self, target, args, kwargs)
            if os.environ.get("TORCH_SPYRE_MOCK_DEBUG_VIEW", "0") == "1" and (
                _contains_mock_spyre_tensor(args) or _contains_mock_spyre_tensor(kwargs)
            ):
                _mock_print(f"[MOCK_DEVICE] interpreter result {_describe_mock_result(result)}")
            return result
        except (RuntimeError, AttributeError, AssertionError) as e:
            msg = str(e)
            is_mock_spyre = _contains_mock_spyre_tensor(args) or _contains_mock_spyre_tensor(kwargs)
            target_name = getattr(target, "__name__", "")
            is_supported_mock_failure = (
                "PyTorch is not linked with support for spyre devices" in msg
                or "'CppOverrides' object has no attribute 'overwrite'" in msg
                or msg == "matmul"
                or "AssertionError: matmul" in msg
                or "batchmatmul: failed to map stick_dims to host coords" in msg
            )
            if not is_supported_mock_failure or not is_mock_spyre:
                raise

            cpu_args = _to_cpu_tree(args)
            cpu_kwargs = _to_cpu_tree(kwargs)

            if target is operator.getitem:
                if MOCK_DEVICE_ENABLED:
                    _mock_print("[MOCK_DEVICE] FX interpreter CPU fallback for builtins.getitem on spyre")
                return target(*cpu_args, **cpu_kwargs)

            if target_name == "overwrite":
                if MOCK_DEVICE_ENABLED:
                    _mock_print("[MOCK_DEVICE] FX interpreter CPU fallback for overwrite on spyre")
                value, stride, offset, _gap = cpu_args
                del stride  # mock fallback only needs logical offset placement
                return value

            if target_name in {"reduction", "_reduction"}:
                if len(cpu_args) >= 4:
                    dtype, src_dtype, reduction_type, value = cpu_args[:4]
                    if reduction_type in {"matmul", "batchmatmul"} and isinstance(value, tuple) and len(value) == 2:
                        if MOCK_DEVICE_ENABLED:
                            _mock_print(
                                f"[MOCK_DEVICE] FX interpreter CPU fallback for reduction({reduction_type}) on spyre"
                            )
                        lhs, rhs = value
                        return lhs * rhs

            target_overloadpacket = getattr(target, "overloadpacket", None)
            target_packet_name = getattr(target_overloadpacket, "__name__", "")
            if target_packet_name == "full":
                if MOCK_DEVICE_ENABLED:
                    _mock_print("[MOCK_DEVICE] FX interpreter CPU fallback for spyre.full on spyre")
                size, fill_value, device, *rest = cpu_args
                dtype = rest[0] if rest else cpu_kwargs.get("dtype")
                return torch.full(size, fill_value, dtype=dtype, device=device)

            if callable(target):
                if MOCK_DEVICE_ENABLED:
                    _mock_print(
                        f"[MOCK_DEVICE] FX interpreter CPU fallback for "
                        f"{getattr(target, '__name__', repr(target))} on spyre"
                    )
                return target(*cpu_args, **cpu_kwargs)

            raise

    def _fake_tensor_torch_function_with_mock_spyre_fallback(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        func_name = getattr(func, "__name__", "")
        flat_args = list(args) + list(kwargs.values())

        if func_name == "copy_":
            if any(_is_mock_spyre_tensor(arg) for arg in flat_args):
                if MOCK_DEVICE_ENABLED:
                    _mock_print("[MOCK_DEVICE] FakeTensor __torch_function__ fallback for copy_ on spyre")
                dst = args[0]
                src = args[1] if len(args) > 1 else None
                try:
                    if src is not None:
                        object.__setattr__(dst, "_mock_spyre_copy_src", src)
                except Exception:
                    pass
                return dst

        if func_name == "contiguous" and args and _is_mock_spyre_tensor(args[0]):
            if MOCK_DEVICE_ENABLED:
                _mock_print("[MOCK_DEVICE] FakeTensor __torch_function__ no-op for contiguous on spyre")
            return args[0]

        if func_name in {"transpose", "permute", "t", "unsqueeze", "squeeze"} and any(
            _is_mock_spyre_tensor(arg) for arg in flat_args
        ):
            assert original_fake_tensor_torch_function is not None
            result = original_fake_tensor_torch_function(func, types, args, kwargs)
            base = args[0] if args else None
            if isinstance(result, FakeTensor) and _is_mock_spyre_tensor(base):
                base_shape = _logical_shape(base)
                try:
                    if func_name == "transpose" and len(args) >= 3 and base_shape is not None:
                        dim0 = args[1]
                        dim1 = args[2]
                        rank = len(base_shape)
                        if dim0 < 0:
                            dim0 += rank
                        if dim1 < 0:
                            dim1 += rank
                        logical_shape = list(base_shape)
                        logical_shape[dim0], logical_shape[dim1] = logical_shape[dim1], logical_shape[dim0]
                        object.__setattr__(result, "_mock_logical_shape", tuple(logical_shape))
                    elif func_name == "t" and base_shape is not None and len(base_shape) == 2:
                        object.__setattr__(result, "_mock_logical_shape", (base_shape[1], base_shape[0]))
                    elif func_name == "unsqueeze" and len(args) >= 2 and base_shape is not None:
                        dim = args[1]
                        rank = len(base_shape) + 1
                        if dim < 0:
                            dim += rank
                        logical_shape = list(base_shape)
                        logical_shape.insert(dim, 1)
                        object.__setattr__(result, "_mock_logical_shape", tuple(logical_shape))
                    elif func_name == "squeeze" and base_shape is not None:
                        if len(args) >= 2:
                            dim = args[1]
                            rank = len(base_shape)
                            if dim < 0:
                                dim += rank
                            logical_shape = list(base_shape)
                            if 0 <= dim < len(logical_shape) and logical_shape[dim] == 1:
                                logical_shape.pop(dim)
                        else:
                            logical_shape = [d for d in base_shape if d != 1]
                        object.__setattr__(result, "_mock_logical_shape", tuple(logical_shape))
                    elif func_name == "permute" and len(args) >= 2 and base_shape is not None:
                        dims = tuple(args[1])
                        object.__setattr__(result, "_mock_logical_shape", tuple(base_shape[i] for i in dims))
                except Exception:
                    pass
            return result

        if func_name in {"mm", "matmul", "bmm"} and any(_is_mock_spyre_tensor(arg) for arg in flat_args):
            if MOCK_DEVICE_ENABLED:
                _mock_print(
                    f"[MOCK_DEVICE] FakeTensor __torch_function__ CPU fallback for {func_name} on spyre"
                )
            cpu_args = _to_cpu_tree(args)
            cpu_kwargs = _to_cpu_tree(kwargs)
            return func(*cpu_args, **cpu_kwargs)

        if str(func) == "aten.bmm.default" and any(_is_mock_spyre_tensor(arg) for arg in flat_args):
            if MOCK_DEVICE_ENABLED:
                _mock_print("[MOCK_DEVICE] FakeTensor __torch_function__ CPU fallback for aten.bmm.default on spyre")
            cpu_args = _to_cpu_tree(args)
            cpu_kwargs = _to_cpu_tree(kwargs)
            return func(*cpu_args, **cpu_kwargs)

        assert original_fake_tensor_torch_function is not None
        return original_fake_tensor_torch_function(func, types, args, kwargs)

    if original_gen_alias_from_base is not None:
        def _gen_alias_from_base_with_mock_spyre_fallback(
            aliased_base_tensor,
            target_meta_tensor,
            target_requires_grad,
            target_view_meta_sequence=None,
            *,
            replay_views,
        ):
            if getattr(getattr(aliased_base_tensor, "device", None), "type", None) == "spyre":
                if replay_views and target_view_meta_sequence is not None:
                    try:
                        replayed = aot_functional_utils._functionalization.apply_view_meta_sequence(
                            aliased_base_tensor,
                            target_view_meta_sequence.sequence,
                        )
                    except Exception:
                        replay_views = False
                    else:
                        if replayed.shape != target_meta_tensor.shape:
                            replay_views = False
            return original_gen_alias_from_base(
                aliased_base_tensor,
                target_meta_tensor,
                target_requires_grad,
                target_view_meta_sequence,
                replay_views=replay_views,
            )

        aot_functional_utils.gen_alias_from_base = _gen_alias_from_base_with_mock_spyre_fallback
        if aot_runtime_wrappers is not None:
            aot_runtime_wrappers.gen_alias_from_base = _gen_alias_from_base_with_mock_spyre_fallback
    dynamo_utils.run_node = _run_node_with_cpu_fallback
    FakeTensor._find_common_device = staticmethod(_find_common_device_with_mock_spyre_support)
    Interpreter.call_function = _interpreter_call_function_with_mock_spyre_fallback
    if original_fake_tensor_torch_function is not None:
        FakeTensor.__torch_function__ = classmethod(_fake_tensor_torch_function_with_mock_spyre_fallback)
    dynamo_utils._spyre_mock_patched = True
    _mock_print("[MOCK_DEVICE] Patched Dynamo fake-tensor run_node for mock spyre backend")


def _patch_cpp_reduction_codegen_for_mock():
    if not MOCK_DEVICE_ENABLED:
        return

    try:
        import torch._inductor.codegen.cpp as cpp_codegen
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Warning: could not import cpp codegen for patching: {e}")
        return

    if getattr(cpp_codegen, "_spyre_mock_matmul_codegen_patched", False):
        return

    original_reduction_combine = cpp_codegen.reduction_combine
    original_reduction_init = cpp_codegen.reduction_init

    def _mock_reduction_combine(reduction_type, a, b, index=None, **kwargs):
        if reduction_type in {"matmul", "batchmatmul"}:
            if isinstance(b, tuple) and len(b) == 2:
                lhs, rhs = b
                return f"({lhs}) * ({rhs})"
            return "0"
        return original_reduction_combine(reduction_type, a, b, index=index, **kwargs)

    def _mock_reduction_init(reduction_type, dtype):
        if reduction_type in {"matmul", "batchmatmul"}:
            return "0"
        return original_reduction_init(reduction_type, dtype)

    cpp_codegen.reduction_combine = _mock_reduction_combine
    cpp_codegen.reduction_init = _mock_reduction_init

    original_gen_parallel_reduction_buffers = cpp_codegen.CppKernel._gen_parallel_reduction_buffers
    original_codegen_functions = cpp_codegen.CppKernelProxy.codegen_functions
    original_kernel_reduction = cpp_codegen.CppKernel.reduction

    def _mock_gen_parallel_reduction_buffers(
        self,
        acc,
        acc_type,
        reduction_type,
        dtype,
        reduction_combine_fn=_mock_reduction_combine,
        reduction_init_fn=_mock_reduction_init,
    ):
        return original_gen_parallel_reduction_buffers(
            self,
            acc,
            acc_type,
            reduction_type,
            dtype,
            reduction_combine_fn=reduction_combine_fn,
            reduction_init_fn=reduction_init_fn,
        )

    def _mock_codegen_functions(self, fn_list, var_sizes_list):
        return original_codegen_functions(self, fn_list, var_sizes_list)

    def _mock_kernel_reduction(self, dtype, src_dtype, reduction_type, value):
        if reduction_type in {"matmul", "batchmatmul"}:
            original_vectorizable_rtypes = cpp_codegen.VECTORIZABLE_RTYPES
            try:
                cpp_codegen.VECTORIZABLE_RTYPES = tuple(
                    list(original_vectorizable_rtypes) + ["matmul", "batchmatmul"]
                )
                return original_kernel_reduction(
                    self, dtype, src_dtype, reduction_type, value
                )
            finally:
                cpp_codegen.VECTORIZABLE_RTYPES = original_vectorizable_rtypes
        return original_kernel_reduction(self, dtype, src_dtype, reduction_type, value)

    cpp_codegen.CppKernel._gen_parallel_reduction_buffers = _mock_gen_parallel_reduction_buffers
    cpp_codegen.CppKernelProxy.codegen_functions = _mock_codegen_functions
    cpp_codegen.CppKernel.reduction = _mock_kernel_reduction
    cpp_codegen._spyre_mock_matmul_codegen_patched = True
    _mock_print("[MOCK_DEVICE] Patched C++ reduction codegen for mock matmul/batchmatmul")


def _patch_spyre_kernel_for_mock():
    """Patch selected Spyre inductor planning logic only in mock mode."""
    if not MOCK_DEVICE_ENABLED:
        return

    try:
        import torch_spyre._inductor.spyre_kernel as spyre_kernel
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Warning: could not import spyre_kernel for patching: {e}")
        return

    if getattr(spyre_kernel, "_spyre_mock_patched", False):
        return

    original_simplify_op_spec = spyre_kernel.simplify_op_spec

    def _simplify_op_spec_with_mock_boundary_fix(op_spec):
        if not op_spec.is_reduction and op_spec.op != "overwrite":
            stick_exprs = set()
            participating_args = 0
            for arg in op_spec.args[:-1]:
                device_coords = getattr(arg, "device_coordinates", None)
                if not device_coords:
                    continue
                stick_expr = device_coords[-1]
                if stick_expr == 0:
                    continue
                free_symbols = getattr(stick_expr, "free_symbols", None)
                if free_symbols is None:
                    participating_args += 1
                    stick_exprs.add(stick_expr)
                    continue
                if len(free_symbols) == 1:
                    participating_args += 1
                    stick_exprs.add(stick_expr)
            if participating_args >= 2 and len(stick_exprs) > 1:
                raise RuntimeError(
                    f"Spyre limitation: pointwise op with nonuniform stick indexing: {stick_exprs}"
                )

        try:
            return original_simplify_op_spec(op_spec)
        except ValueError as e:
            if "is not in list" not in str(e):
                raise

            try:
                import torch_spyre._inductor.views as spyre_views
            except Exception:
                raise

            var_ranges = {var: val[0] for var, val in op_spec.iteration_space.items()}
            op_it_space_splits = {var: val[1] for var, val in op_spec.iteration_space.items()}

            splits = {var: set() for var in var_ranges.keys()}
            breakdown = []
            stick_dim = []
            stick_size = []
            for arg in op_spec.args:
                tensor = {
                    "size": arg.device_size,
                    "coordinates": arg.device_coordinates,
                }
                intervals = spyre_views.normalize_coordinates(
                    var_ranges, tensor["size"], tensor["coordinates"]
                )
                stick_dim.append(intervals[-1][2])
                stick_size.append(intervals[-1][-1])
                breakdown.append(intervals)
                for num, den, var, mod, dim_size in intervals:
                    if var is not None:
                        if den != stick_size[-1] or var != stick_dim[-1]:
                            splits[var].add(den)
                        if mod != stick_size[-1] or var != stick_dim[-1]:
                            splits[var].add(mod)

            splits = {
                var: sorted({sympy.S.One, var_ranges[var], *val})
                for var, val in splits.items()
            }

            new_var_ranges = {}
            new_op_it_space_splits = {}
            n = 0
            remap = {}
            for var, split in splits.items():
                div = op_it_space_splits[var] if var in op_it_space_splits else 1
                if len(split) > 1:
                    new_var_ranges[var] = split[1] // split[0]
                    remap[var] = [var]
                    for i in range(1, len(split) - 1):
                        new_var = sympy.symbols(f"z{n}")
                        n += 1
                        new_var_ranges[new_var] = split[i + 1] // split[i]
                        remap[var].append(new_var)
                    for v in reversed(remap[var]):
                        new_op_it_space_splits[v] = math.gcd(div, new_var_ranges[v])
                        div //= new_op_it_space_splits[v]
                else:
                    new_var_ranges[var] = var_ranges[var]
                    new_op_it_space_splits[var] = (
                        op_it_space_splits[var] if var in op_it_space_splits else 1
                    )

            new_tensors = []
            for j, intervals in enumerate(breakdown):
                size = []
                coordinates = []
                for num, den, var, mod, dim_size in intervals[:-1]:
                    if var is None:
                        size.append(dim_size)
                        coordinates.append(sympy.S.Zero)
                        continue
                    low = (
                        0
                        if var == stick_dim[j]
                        and den == stick_size[j]
                        and den not in splits[var]
                        else splits[var].index(den)
                    )
                    for i in reversed(range(low, splits[var].index(mod))):
                        if i == splits[var].index(mod) - 1:
                            size.append(dim_size * den // splits[var][i])
                        else:
                            size.append(splits[var][i + 1] // splits[var][i])
                        coordinates.append(remap[var][i])
                    if var == stick_dim[j] and den == stick_size[j] and den not in splits[var]:
                        size[-1] //= den
                        coordinates[-1] //= den
                    if num > 1:
                        size.append(num)
                        coordinates.append(sympy.S.Zero)
                num, den, var, mod, dim_size = intervals[-1]
                size.append(dim_size)
                coordinates.append(var % dim_size if var is not None else sympy.S.Zero)
                new_tensors.append({"size": size, "coordinates": coordinates})

            rank = 0
            for i, t in enumerate(new_tensors):
                if stick_dim[i] is None:
                    not_found = 1
                    for c, s in zip(t["coordinates"][:-1], t["size"][:-1]):
                        if c == 0 and s == 1:
                            not_found = 0
                            break
                    rank = max(rank, len(t["size"]) + not_found)
                    continue
                found = 1
                for c, s in zip(t["coordinates"][:-1], t["size"][:-1]):
                    if stick_dim[i] in c.free_symbols or s == 1:
                        found = 0
                        break
                rank = max(rank, len(t["size"]) + found)

            for t in new_tensors:
                gap = rank - len(t["size"])
                t["size"] = [sympy.S.One] * gap + t["size"]
                t["coordinates"] = [sympy.S.Zero] * gap + t["coordinates"]

            for t in new_tensors:
                vars = t["coordinates"][-1].free_symbols
                if len(vars) == 1:
                    stick_dim_var = next(iter(vars))
                    found = False
                    for i in range(len(t["coordinates"]) - 1):
                        vars = t["coordinates"][i].free_symbols
                        if stick_dim_var in vars:
                            found = True
                            continue
                    if not found:
                        for i in range(len(t["coordinates"]) - 1):
                            if t["size"][i] == 1:
                                t["coordinates"][i] = stick_dim_var // t["size"][-1]
                                t["coordinates"][-1] = stick_dim_var % t["size"][-1]
                                break

            op_spec.iteration_space = {
                k: (v, new_op_it_space_splits[k]) for k, v in new_var_ranges.items()
            }
            for arg, t in zip(op_spec.args, new_tensors):
                arg.device_size = t["size"]
                arg.device_coordinates = t["coordinates"]

            if MOCK_DEVICE_ENABLED:
                _mock_print("[MOCK_DEVICE] Applied mock-only Spyre planner boundary fix")
            return None

    spyre_kernel.simplify_op_spec = _simplify_op_spec_with_mock_boundary_fix
    setattr(spyre_kernel, "_spyre_mock_patched", True)
    _mock_print("[MOCK_DEVICE] Patched spyre_kernel.simplify_op_spec for mock spyre backend")


def start_runtime():
    """Stub function - initializes mock device"""
    if MOCK_DEVICE_ENABLED:
        _mock_print("[MOCK_DEVICE] Runtime initialized")
        _register_privateuse1_backend()
        _patch_torch_accelerator_for_mock()
        _patch_fake_tensor_ops_for_mock()
        _patch_cpp_reduction_codegen_for_mock()
        _patch_spyre_kernel_for_mock()
    pass


def _register_privateuse1_backend():
    """Register Spyre as PrivateUse1 backend with PyTorch"""
    if not MOCK_DEVICE_ENABLED:
        return
    
    try:
        import torch
        # Register the backend name
        torch._register_device_module('spyre', 'torch_spyre')
        _mock_print("[MOCK_DEVICE] Registered 'spyre' as PrivateUse1 backend")
    except Exception as e:
        _mock_print(f"[MOCK_DEVICE] Warning: Could not register backend: {e}")


def get_device_dtype(dtype):
    """Stub function - maps torch dtype to DataFormats"""
    import torch
    if dtype == torch.float16:
        return DataFormats.SEN169_FP16
    elif dtype == torch.float32:
        return DataFormats.IEEE_FP32
    else:
        # Default to FP16 for other types
        return DataFormats.SEN169_FP16


def get_elem_in_stick(dtype):
    """Stub function - returns number of elements in a stick"""
    return 1  # Return 1 as default


# Mock device registration functions
def is_available():
    """Mock device is always available when enabled"""
    return MOCK_DEVICE_ENABLED


def device_count():
    """Return 1 mock device"""
    return 1 if MOCK_DEVICE_ENABLED else 0


def current_device():
    """Return current device (always 0 for mock)"""
    return 0


def set_device(device_id):
    """Set current device (no-op for mock)"""
    if MOCK_DEVICE_ENABLED:
        _mock_print(f"[MOCK_DEVICE] set_device({device_id}) - no-op")
    pass


def _register_device():
    """Register device (no-op for mock - already registered via _register_privateuse1_backend)"""
    if MOCK_DEVICE_ENABLED:
        _mock_print("[MOCK_DEVICE] _register_device() called - no-op")
    return True


def _is_device_available():
    """Check if device is available"""
    return MOCK_DEVICE_ENABLED


def _get_device_count():
    """Get device count"""
    return 1 if MOCK_DEVICE_ENABLED else 0


# Mock tensor storage - keeps data on CPU but tracks device metadata
class MockSpyreTensorStorage:
    """Mock storage that keeps tensor data on CPU but tracks Spyre device metadata"""
    def __init__(self, cpu_tensor):
        self.cpu_data = cpu_tensor
        self.device_id = 0
        self.is_mock = True
        if MOCK_DEVICE_ENABLED:
            _mock_print(f"[MOCK_DEVICE] Created mock tensor storage: shape={cpu_tensor.shape}, dtype={cpu_tensor.dtype}")


# Add any other functions/classes that might be imported from _C
__all__ = [
    'DataFormats',
    'SpyreTensorLayout',
    'get_spyre_tensor_layout',
    'to_with_layout',
    'empty_with_layout',
    'spyre_empty_with_layout',
    'reinterpret_tensor',
    'reinterpret_tensor_with_layout',
    'encode_constant',
    'convert_artifacts',
    'launch_kernel',
    'start_runtime',
    'get_device_dtype',
    'get_elem_in_stick',
    '_register_device',
    '_is_device_available',
    '_get_device_count',
    'MockSpyreTensorStorage',
    'MOCK_DEVICE_ENABLED',
]
