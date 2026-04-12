"""Mock device module for torch-spyre.

This module provides a mock implementation of the Spyre device that allows
testing and development without requiring actual hardware.

Architecture:
    core/           - Generic framework (device-agnostic abstractions)
    backends/spyre/ - Spyre-specific implementation
"""

# Configuration and logging
from .config import MockDeviceConfig, is_mock_enabled
from .logger import get_logger

# Core abstractions
from .core import (
    TensorDescriptor,
    ComputeOp,
    MockOpSpec,
    AbstractGraphValidator,
    BaseGraphValidator,
    AbstractLayoutTransformer,
    MockOpDispatcher,
    AbstractMockDevice,
    OpRegistry,
    DTypeRegistry,
)

# Spyre backend
from .backends.spyre import (
    MockSpyreDevice,
    SpyreSDSCMockKernelRunner,
)

__all__ = [
    # Configuration
    "MockDeviceConfig",
    "is_mock_enabled",
    "get_logger",
    
    # Core abstractions
    "TensorDescriptor",
    "ComputeOp",
    "MockOpSpec",
    "AbstractGraphValidator",
    "BaseGraphValidator",
    "AbstractLayoutTransformer",
    "MockOpDispatcher",
    "AbstractMockDevice",
    "OpRegistry",
    "DTypeRegistry",
    
    # Spyre backend
    "MockSpyreDevice",
    "SpyreSDSCMockKernelRunner",
]
