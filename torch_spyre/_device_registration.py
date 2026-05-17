"""
Minimal device registration for mock mode
"""
import os
import torch


def _stub_copy_impl(self, src, non_blocking=False):
    """Stub implementation for copy_ operation"""
    # For mock device, just copy the data on CPU
    if hasattr(src, '_mock_cpu_tensor'):
        self.copy_(src._mock_cpu_tensor, non_blocking=non_blocking)
    else:
        self.copy_(src, non_blocking=non_blocking)
    return self


def register_mock_spyre_device():
    """Register spyre as a PrivateUse1 device when in mock mode"""
    mock_enabled = os.environ.get('TORCH_SPYRE_MOCK_DEVICE', '0') == '1'
    if not mock_enabled:
        return False
    
    try:
        # Check if the function exists (PyTorch 1.13+)
        if not hasattr(torch.utils, 'rename_privateuse1_backend'):
            print("[MOCK_DEVICE] Warning: rename_privateuse1_backend not available (requires PyTorch 1.13+)")
            return False
        
        # Try to rename PrivateUse1 to 'spyre'
        # This allows torch.device("spyre") to work
        torch.utils.rename_privateuse1_backend("spyre")  # type: ignore
        print("[MOCK_DEVICE] Registered 'spyre' as PrivateUse1 backend")
        
        # Register stub generator for the backend
        # This tells PyTorch how to create tensors on the spyre device
        def spyre_generator(device_idx):
            """Return CPU generator for mock device"""
            return torch.Generator(device='cpu')
        
        # Try to register the generator (may not be available in all PyTorch versions)
        if hasattr(torch.utils, 'generate_methods_for_privateuse1_backend'):
            try:
                torch.utils.generate_methods_for_privateuse1_backend()  # type: ignore
                print("[MOCK_DEVICE] Generated methods for spyre backend")
            except Exception as e:
                print(f"[MOCK_DEVICE] Note: Could not generate methods: {e}")
        
        return True
        
    except Exception as e:
        print(f"[MOCK_DEVICE] Could not register device: {e}")
        return False
