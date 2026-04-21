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

MOCK_DEVICE_ENABLED = os.environ.get('TORCH_SPYRE_MOCK_DEVICE', '0') == '1'


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
            print(f"[MOCK_DEVICE] Created MockSpyreTensor: shape={self.shape}, dtype={self.dtype}")
    
    @property
    def device(self):
        """Report device as 'spyre' to trick inductor"""
        # Handle case where _mock_device might not be set (e.g., after copy_())
        if not hasattr(self, '_mock_device'):
            self._mock_device = torch.device("spyre", 0)
        return self._mock_device
    
    def device_tensor_layout(self):
        """
        Return a mock device tensor layout.
        This is required by Spyre inductor backend.
        """
        if MOCK_DEVICE_ENABLED:
            print(f"[MOCK_DEVICE] device_tensor_layout() called for tensor shape={self.shape}")
        
        # Import here to avoid circular dependency
        try:
            from torch_spyre._C import SpyreTensorLayout, DataFormats, get_device_dtype
            
            # Create layout with tensor's actual dimensions
            device_size = tuple(self.shape)
            
            # Create simple dim_map and stride_map based on tensor layout
            dim_map = {i: i for i in range(len(self.shape))}
            stride_map = {i: int(s) for i, s in enumerate(self.stride())}
            
            # Get appropriate data format for dtype
            try:
                data_format = get_device_dtype(self.dtype)
            except:
                data_format = DataFormats.IEEE_FP32
            
            layout = SpyreTensorLayout(
                device_size=device_size,
                dim_map=dim_map,
                stride_map=stride_map,
                data_format=data_format
            )
            
            if MOCK_DEVICE_ENABLED:
                print(f"[MOCK_DEVICE] Created SpyreTensorLayout: device_size={device_size}")
            return layout
        except Exception as e:
            if MOCK_DEVICE_ENABLED:
                print(f"[MOCK_DEVICE] Warning: Could not create SpyreTensorLayout: {e}")
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
                print(f"[MOCK_DEVICE] Tensor already on spyre device")
            return self
        
        # Otherwise, convert back to regular tensor and transfer
        if MOCK_DEVICE_ENABLED:
            print(f"[MOCK_DEVICE] Converting MockSpyreTensor to regular tensor for device: {device}")
        regular_tensor = torch.Tensor._make_subclass(torch.Tensor, self)
        return regular_tensor.to(*args, **kwargs)
    
    def cpu(self):
        """Return as regular CPU tensor"""
        if MOCK_DEVICE_ENABLED:
            print(f"[MOCK_DEVICE] Converting MockSpyreTensor to CPU tensor")
        return torch.Tensor._make_subclass(torch.Tensor, self)
    
    def __repr__(self):
        return f"MockSpyreTensor({super().__repr__()}, device='{self.device}')"


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
        print(f"[MOCK_DEVICE] Converting tensor to mock Spyre device")
    
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
                print(f"[MOCK_DEVICE] Intercepted .to('spyre') call")
            return to_mock_spyre(self)
        
        # Otherwise use original implementation
        return _original_tensor_to(self, *args, **kwargs)
    except RuntimeError as e:
        # If PyTorch rejects the device, check if it's spyre and handle it
        if 'spyre' in str(e).lower() or 'privateuse1' in str(e).lower():
            if MOCK_DEVICE_ENABLED:
                print(f"[MOCK_DEVICE] Caught PyTorch device error, converting to MockSpyreTensor")
            return to_mock_spyre(self)
        raise


def install_mock_device_patches():
    """Install patches to enable mock Spyre device"""
    if not MOCK_DEVICE_ENABLED:
        return
    
    print("[MOCK_DEVICE] Installing tensor.to() patch for Spyre device")
    torch.Tensor.to = _patched_tensor_to  # type: ignore
    print("[MOCK_DEVICE] Mock device patches installed successfully")


# Auto-install if mock device is enabled
if MOCK_DEVICE_ENABLED:
    install_mock_device_patches()
