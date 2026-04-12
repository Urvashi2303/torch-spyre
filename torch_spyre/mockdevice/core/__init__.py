"""Core abstractions for mock device implementation."""

from .graph import TensorDescriptor, ComputeOp, MockOpSpec
from .interfaces import (
    AbstractGraphValidator,
    BaseGraphValidator,
    AbstractLayoutTransformer,
    MockOpDispatcher,
    AbstractMockDevice,
)
from .registry import OpRegistry, DTypeRegistry
from .serialization import OpSpecSerializer, save_opspec, load_opspec

__all__ = [
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
    "OpSpecSerializer",
    "save_opspec",
    "load_opspec",
]
