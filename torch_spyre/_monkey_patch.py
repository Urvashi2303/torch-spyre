# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import os

# Try to import from _C, but allow graceful failure
try:
    from torch_spyre._C import get_spyre_tensor_layout, to_with_layout, empty_with_layout
    from torch_spyre._C import SpyreTensorLayout
    _C_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Stub implementations when _C is not available
    get_spyre_tensor_layout = None
    to_with_layout = None
    empty_with_layout = None
    SpyreTensorLayout = None
    _C_AVAILABLE = False


MOCK_VERBOSE = os.environ.get("TORCH_SPYRE_MOCK_VERBOSE", "0") == "1"


def _mock_print(*args, **kwargs):
    if MOCK_VERBOSE:
        print(*args, **kwargs)


def _patch_tensor_for_spyre():
    import torch

    def _patch_mock_stream_api():
        if not mock_device_enabled:
            return
        if getattr(torch, "_spyre_mock_stream_patched", False):
            return

        def _mock_torch_stream(device=None, priority=0, **kwargs):
            del kwargs
            from torch_spyre.streams import Stream
            return Stream(device=device, priority=priority)

        torch.Stream = _mock_torch_stream
        torch._spyre_mock_stream_patched = True

    if getattr(torch.Tensor, "_spyre_tensor_patched", False):
        _patch_mock_stream_api()
        return
    
    # Check if mock device mode is enabled
    mock_device_enabled = os.environ.get('TORCH_SPYRE_MOCK_DEVICE', '0') == '1'
    
    _patch_mock_stream_api()

    # Skip patching if _C module is not available and mock device is not enabled
    if not _C_AVAILABLE and not mock_device_enabled:
        import warnings
        warnings.warn(
            "Skipping tensor patching: torch_spyre._C not available and mock device not enabled. "
            "Set TORCH_SPYRE_MOCK_DEVICE=1 to enable mock device mode.",
            RuntimeWarning,
            stacklevel=2
        )
        torch.Tensor._spyre_tensor_patched = True
        return

    orig_repr = torch.Tensor.__repr__
    orig_to = torch.Tensor.to
    orig_empty = torch.empty
    orig_tensor = torch.tensor
    orig_zeros = torch.zeros
    orig_ones = torch.ones
    orig_full = torch.full
    orig_randn = torch.randn

    def spyre_aware_repr(self):
        dev = getattr(self, "device", None)
        if dev is not None and dev.type == "spyre":
            try:
                s = orig_repr(self.to("cpu"))
            except Exception:
                # Fallback if .to("cpu") fails for some weird reason
                return (
                    f"SpyreTensor(shape={tuple(self.shape)}, "
                    f"dtype={self.dtype}, device={self.device})"
                )
            if "device=" in s:
                return s.replace("device='cpu'", f"device='{self.device}'")
            if s.endswith(")"):
                s = s[:-1] + f", device='{self.device}')"
            else:
                # Odd case: just append device info
                s = s + f" (device='{self.device}')"
            return s

        # Non-spyre tensors use normal behavior
        return orig_repr(self)

    def device_tensor_layout(self: torch.Tensor) -> Optional[SpyreTensorLayout]:
        if self.device is not None and self.device.type == "spyre":
            return get_spyre_tensor_layout(self)
        else:
            return None

    def spyre_to(self, *args, device_layout=None, **kwargs):
        # Check if trying to move to spyre device
        target_device = None
        
        # Extract target device from args or kwargs
        if args:
            first_arg = args[0]
            if isinstance(first_arg, torch.device):
                target_device = first_arg.type
            elif isinstance(first_arg, str):
                target_device = first_arg
        
        if 'device' in kwargs:
            device_arg = kwargs['device']
            if isinstance(device_arg, torch.device):
                target_device = device_arg.type
            elif isinstance(device_arg, str):
                target_device = device_arg
        
        # If moving to spyre device and mock mode is enabled, use MockSpyreTensor
        # This must happen BEFORE calling orig_to to avoid C++ validation
        if target_device and 'spyre' in target_device and mock_device_enabled:
            try:
                from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor
                _mock_print(f"[MOCK_DEVICE] Intercepting .to('spyre') - converting to MockSpyreTensor")
                return MockSpyreTensor(self)
            except ImportError as e:
                import warnings
                warnings.warn(f"Could not import MockSpyreTensor: {e}", RuntimeWarning)
                # Fall through to original behavior
        
        # Preserve explicit device_layout in mock mode even when no explicit spyre
        # target device string is passed (e.g. x.to(device_layout=layout)).
        if device_layout is not None and mock_device_enabled:
            try:
                from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor
                from torch_spyre._C import get_spyre_tensor_layout

                base = self if isinstance(self, MockSpyreTensor) else MockSpyreTensor(self)
                setattr(base, "_mock_device_layout", device_layout)
                return base
            except ImportError as e:
                import warnings
                warnings.warn(f"Could not preserve mock device layout: {e}", RuntimeWarning)

        # Original behavior for non-spyre devices or when _C is available
        if device_layout is None:
            return orig_to(self, *args, **kwargs)
        else:
            if _C_AVAILABLE:
                return to_with_layout(self, device_layout)
            else:
                return orig_to(self, *args, **kwargs)

    def spyre_tensor(data, *args, **kwargs):
        if mock_device_enabled:
            device = kwargs.get("device", None)
            device_type = None
            if isinstance(device, str):
                device_type = device.split(":", 1)[0]
            else:
                device_type = getattr(device, "type", None)

            if device_type == "spyre":
                kwargs = dict(kwargs)
                kwargs["device"] = "cpu"
                cpu_tensor = orig_tensor(data, *args, **kwargs)
                if type(cpu_tensor).__name__ == "FakeTensor" or getattr(
                    getattr(cpu_tensor, "device", None), "type", None
                ) == "meta":
                    return cpu_tensor
                from torch_spyre.mock_device_integration.mock_spyre_tensor import (
                    MockSpyreTensor,
                )

                return MockSpyreTensor(cpu_tensor)

        return orig_tensor(data, *args, **kwargs)

    def _mock_wrap_factory_result(result):
        if type(result).__name__ == "FakeTensor" or getattr(
            getattr(result, "device", None), "type", None
        ) == "meta":
            return result
        from torch_spyre.mock_device_integration.mock_spyre_tensor import MockSpyreTensor
        return MockSpyreTensor(result)

    def _normalize_mock_factory_device(device):
        if isinstance(device, str):
            return device.split(":", 1)[0]
        return getattr(device, "type", None)

    def spyre_empty(
        *args,
        device_layout=None,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
        memory_format=torch.contiguous_format,
    ):
        if device_layout is None:  # use original implementation if no layout is provided
            if mock_device_enabled and device is not None:
                device_type = None
                if isinstance(device, str):
                    device_type = device.split(":", 1)[0]
                else:
                    device_type = getattr(device, "type", None)

                if device_type == "spyre":
                    cpu_tensor = orig_empty(
                        *args,
                        out=None,
                        dtype=dtype,
                        layout=layout,
                        device="cpu",
                        requires_grad=requires_grad,
                        pin_memory=pin_memory,
                        memory_format=memory_format,
                    )
                    if type(cpu_tensor).__name__ == "FakeTensor" or getattr(
                        getattr(cpu_tensor, "device", None), "type", None
                    ) == "meta":
                        return cpu_tensor
                    from torch_spyre.mock_device_integration.mock_spyre_tensor import (
                        MockSpyreTensor,
                    )

                    return MockSpyreTensor(cpu_tensor)

            return orig_empty(
                *args,
                out=out,
                dtype=dtype,
                layout=layout,
                device=device,
                requires_grad=requires_grad,
                pin_memory=pin_memory,
                memory_format=memory_format,
            )
        else:
            # layout_opt is omitted; c10::Layout has no pybind11 type caster,
            # so py_empty_with_layout drops that parameter and always uses
            # the default (Strided).
            return empty_with_layout(
                *args, device_layout, dtype, device, pin_memory, memory_format
            )

    def spyre_zeros(*args, **kwargs):
        if mock_device_enabled and _normalize_mock_factory_device(kwargs.get("device")) == "spyre":
            kwargs = dict(kwargs)
            kwargs["device"] = "cpu"
            return _mock_wrap_factory_result(orig_zeros(*args, **kwargs))
        return orig_zeros(*args, **kwargs)

    def spyre_ones(*args, **kwargs):
        if mock_device_enabled and _normalize_mock_factory_device(kwargs.get("device")) == "spyre":
            kwargs = dict(kwargs)
            kwargs["device"] = "cpu"
            return _mock_wrap_factory_result(orig_ones(*args, **kwargs))
        return orig_ones(*args, **kwargs)

    def spyre_full(*args, **kwargs):
        if mock_device_enabled:
            normalized_device = _normalize_mock_factory_device(kwargs.get("device"))
            if normalized_device == "spyre":
                kwargs = dict(kwargs)
                kwargs["device"] = "cpu"
                return _mock_wrap_factory_result(orig_full(*args, **kwargs))
            if normalized_device == "cpu":
                return orig_full(*args, **kwargs)
        return orig_full(*args, **kwargs)

    def spyre_randn(*args, **kwargs):
        if mock_device_enabled and _normalize_mock_factory_device(kwargs.get("device")) == "spyre":
            kwargs = dict(kwargs)
            kwargs["device"] = "cpu"
            return _mock_wrap_factory_result(orig_randn(*args, **kwargs))
        return orig_randn(*args, **kwargs)

    orig_empty_like = torch.empty_like

    def spyre_empty_like(*args, **kwargs):
        if mock_device_enabled and _normalize_mock_factory_device(kwargs.get("device")) == "spyre":
            kwargs = dict(kwargs)
            kwargs["device"] = "cpu"
            return _mock_wrap_factory_result(orig_empty_like(*args, **kwargs))
        return orig_empty_like(*args, **kwargs)

    torch.Tensor.__repr__ = spyre_aware_repr
    torch.Tensor.device_tensor_layout = device_tensor_layout
    torch.Tensor._spyre_tensor_patched = True
    torch.Tensor.to = spyre_to
    torch.empty = spyre_empty
    torch.empty_like = spyre_empty_like
    torch.tensor = spyre_tensor
    torch.zeros = spyre_zeros
    torch.ones = spyre_ones
    torch.full = spyre_full
    torch.randn = spyre_randn


def _patch_torch_device():
    """Patch torch.device to handle 'spyre' device when in mock mode"""
    import torch
    import os
    
    if getattr(torch.device, "_spyre_device_patched", False):
        return
    
    mock_device_enabled = os.environ.get('TORCH_SPYRE_MOCK_DEVICE', '0') == '1'
    if not mock_device_enabled:
        return
    
    # Store original torch.device
    _original_torch_device = torch.device
    
    class SpyreDeviceWrapper:
        """Wrapper that allows torch.device('spyre') to work in mock mode"""
        def __new__(cls, device):
            # If it's a spyre device string, create a CPU device but mark it
            if isinstance(device, str) and 'spyre' in device:
                _mock_print("[MOCK_DEVICE] torch.device('spyre') called - creating mock device")
                # Return a regular CPU device - the .to() method will handle conversion
                return _original_torch_device('cpu')
            elif isinstance(device, torch.device) and device.type == 'spyre':
                return _original_torch_device('cpu')
            else:
                return _original_torch_device(device)
    
    # Replace torch.device with our wrapper
    torch.device = SpyreDeviceWrapper
    torch.device._spyre_device_patched = True
    _mock_print("[MOCK_DEVICE] torch.device patched to handle 'spyre' device")
