# Copyright 2025 The Torch-Spyre Authors.
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

# Check if mock device is enabled
MOCK_DEVICE_ENABLED = os.environ.get('TORCH_SPYRE_MOCK_DEVICE', '0') == '1'

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
    print("[MOCK_DEVICE] Spyre mock device enabled - SDSC JSON generation active")


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
DataFormats.SEN169_FP16 = DataFormats(0, "SEN169_FP16", elems=32)  # FP16 typically has 32 elements per stick
DataFormats.IEEE_FP32 = DataFormats(1, "IEEE_FP32", elems=16)      # FP32 typically has 16 elements per stick
DataFormats.SEN169_FP8 = DataFormats(2, "SEN169_FP8", elems=64)    # FP8 typically has 64 elements per stick


class SpyreTensorLayout:
    """Stub SpyreTensorLayout class with required attributes for inductor"""
    def __init__(self, device_size=None, dim_map=None, stride_map=None, data_format=None):
        # device_size: tuple of ints representing tensor size on device
        # dim_map: mapping of dimensions
        # stride_map: mapping of strides
        # data_format: DataFormats enum value
        self.device_size = device_size if device_size is not None else ()
        self.dim_map = dim_map if dim_map is not None else {}
        self.stride_map = stride_map if stride_map is not None else {}
        self.data_format = data_format if data_format is not None else DataFormats.IEEE_FP32
        self.device_dtype = data_format if data_format is not None else DataFormats.IEEE_FP32
        
        if MOCK_DEVICE_ENABLED:
            print(f"[MOCK_DEVICE] SpyreTensorLayout created: device_size={self.device_size}")
    
    def elems_per_stick(self):
        """Return number of elements per stick (default 1 for mock)"""
        return 1
    
    def __repr__(self):
        """Return a valid Python expression that can recreate this object"""
        return (
            f"SpyreTensorLayout("
            f"device_size={self.device_size!r}, "
            f"dim_map={self.dim_map!r}, "
            f"stride_map={self.stride_map!r}, "
            f"data_format={self.data_format!r})"
        )


def get_spyre_tensor_layout(tensor):
    """Stub function"""
    return None


def to_with_layout(tensor, layout):
    """Stub function"""
    return tensor


def empty_with_layout(*args, **kwargs):
    """Stub function"""
    import torch
    return torch.empty(*args[:1])  # Just create a regular empty tensor


def spyre_empty_with_layout(*args, **kwargs):
    """Stub function"""
    import torch
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


def start_runtime():
    """Stub function - initializes mock device"""
    if MOCK_DEVICE_ENABLED:
        print("[MOCK_DEVICE] Runtime initialized")
        _register_privateuse1_backend()
    pass


def _register_privateuse1_backend():
    """Register Spyre as PrivateUse1 backend with PyTorch"""
    if not MOCK_DEVICE_ENABLED:
        return
    
    try:
        import torch
        # Register the backend name
        torch._register_device_module('spyre', 'torch_spyre')
        print("[MOCK_DEVICE] Registered 'spyre' as PrivateUse1 backend")
    except Exception as e:
        print(f"[MOCK_DEVICE] Warning: Could not register backend: {e}")


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
        print(f"[MOCK_DEVICE] set_device({device_id}) - no-op")
    pass


def _register_device():
    """Register device (no-op for mock - already registered via _register_privateuse1_backend)"""
    if MOCK_DEVICE_ENABLED:
        print("[MOCK_DEVICE] _register_device() called - no-op")
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
            print(f"[MOCK_DEVICE] Created mock tensor storage: shape={cpu_tensor.shape}, dtype={cpu_tensor.dtype}")


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
