# Copyright 2025 The Torch-Spyre Authors.
# Stub implementation of _hooks module for when C++ extensions are not available

"""
Spyre bootstrap: registers PrivateUse1 hooks.
This stub version provides minimal functionality for mock device mode.
"""

import os
import warnings

MOCK_DEVICE_ENABLED = os.environ.get('TORCH_SPYRE_MOCK_DEVICE', '0') == '1'
MOCK_VERBOSE = os.environ.get("TORCH_SPYRE_MOCK_VERBOSE", "0") == "1"

if MOCK_DEVICE_ENABLED and MOCK_VERBOSE:
    print("[MOCK_DEVICE] Loading stub _hooks module")

# Stub implementation - no actual hooks registered
# The real _hooks.so would register C++ hooks with PyTorch
# In mock mode, we rely on Python-level patching instead

__all__ = []
