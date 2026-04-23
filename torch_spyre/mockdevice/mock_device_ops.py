# Copyright 2025 The Torch-Spyre Authors.
# Mock device operations for SDSC JSON generation

"""
Mock device operations that intercept tensor.to("spyre") calls
and enable SDSC JSON generation without actual hardware.
"""

import torch
import os
import json
from typing import Any, Dict, List

MOCK_DEVICE_ENABLED = os.environ.get('TORCH_SPYRE_MOCK_DEVICE', '0') == '1'

# Global storage for tracking operations
_operation_log = []
_sdsc_graph = {
    "version": "1.0",
    "operations": [],
    "tensors": {}
}


def log_operation(op_name: str, inputs: List[Any], output: Any, metadata: Dict[str, Any] | None = None):
    """Log an operation for SDSC JSON generation"""
    if not MOCK_DEVICE_ENABLED:
        return
    
    op_entry = {
        "name": op_name,
        "inputs": [_tensor_info(inp) for inp in inputs if isinstance(inp, torch.Tensor)],
        "output": _tensor_info(output) if isinstance(output, torch.Tensor) else None,
        "metadata": metadata if metadata is not None else {}
    }
    _operation_log.append(op_entry)
    _sdsc_graph["operations"].append(op_entry)
    
    if MOCK_DEVICE_ENABLED:
        print(f"[MOCK_DEVICE] Logged operation: {op_name}")


def _tensor_info(tensor):
    """Extract tensor information for SDSC"""
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad
    }


def get_sdsc_json():
    """Get the accumulated SDSC JSON"""
    return json.dumps(_sdsc_graph, indent=2)


def clear_sdsc_log():
    """Clear the operation log"""
    global _operation_log, _sdsc_graph
    _operation_log = []
    _sdsc_graph = {
        "version": "1.0",
        "operations": [],
        "tensors": {}
    }


def print_sdsc_json():
    """Print the SDSC JSON"""
    print("\n" + "="*60)
    print("SDSC JSON OUTPUT")
    print("="*60)
    print(get_sdsc_json())
    print("="*60 + "\n")


# Mock tensor.to() implementation
class MockSpyreTensor:
    """Wrapper that keeps tensor on CPU but tracks as Spyre device"""
    
    def __init__(self, cpu_tensor):
        self._cpu_tensor = cpu_tensor
        self._device_str = "spyre:0"
        if MOCK_DEVICE_ENABLED:
            print(f"[MOCK_DEVICE] Created mock Spyre tensor: shape={cpu_tensor.shape}, dtype={cpu_tensor.dtype}")
    
    @property
    def device(self):
        return torch.device(self._device_str)
    
    def cpu(self):
        """Return the underlying CPU tensor"""
        if MOCK_DEVICE_ENABLED:
            print(f"[MOCK_DEVICE] Moving tensor back to CPU")
        return self._cpu_tensor
    
    def __getattr__(self, name):
        """Forward all other attributes to CPU tensor"""
        return getattr(self._cpu_tensor, name)


def mock_tensor_to(tensor, device, *args, **kwargs):
    """Mock implementation of tensor.to() for Spyre device"""
    if isinstance(device, str) and 'spyre' in device.lower():
        if MOCK_DEVICE_ENABLED:
            print(f"[MOCK_DEVICE] Intercepted tensor.to('spyre')")
        # Keep tensor on CPU but mark as Spyre device
        return MockSpyreTensor(tensor)
    # For other devices, use original implementation
    return tensor.to(device, *args, **kwargs)


def install_mock_device_hooks():
    """Install hooks to intercept device operations"""
    if not MOCK_DEVICE_ENABLED:
        return
    
    print("[MOCK_DEVICE] Installing device operation hooks")
    
    # We'll patch torch.Tensor.to to intercept device transfers
    # This is a simplified approach - full implementation would need more hooks
    original_to = torch.Tensor.to
    
    def patched_to(self, *args, **kwargs):
        # Check if first arg is a device string containing 'spyre'
        if args and isinstance(args[0], (str, torch.device)):
            device_str = str(args[0])
            if 'spyre' in device_str.lower():
                return mock_tensor_to(self, args[0], *args[1:], **kwargs)  # type: ignore
        return original_to(self, *args, **kwargs)
    
    torch.Tensor.to = patched_to  # type: ignore
    print("[MOCK_DEVICE] Hooks installed successfully")


# Initialize hooks on import if mock device is enabled
if MOCK_DEVICE_ENABLED:
    install_mock_device_hooks()
