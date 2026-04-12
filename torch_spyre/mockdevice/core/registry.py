"""
torch_spyre.mockdevice.core.registry
------------------------------------
Simple registries for operations and data types

This file contains two simple "dictionary-like" classes:
1. OpRegistry: Stores custom operation implementations
2. DTypeRegistry: Maps device format strings to PyTorch data types
"""
from __future__ import annotations
from typing import Callable, Optional
import torch
from ..logger import get_logger

logger = get_logger("Registry")


class OpRegistry:
    """
    A simple registry that stores custom operation implementations.
    
    Think of it like a phonebook: you register an operation name with a function,
    and later you can look up that function by name.
    
    Example:
        registry = OpRegistry("MyDevice")
        
        @registry.register("my_add")
        def custom_add(args, op, verbose):
            return [args[0] + args[1]]
        
        # Later, look it up:
        fn = registry.lookup("my_add")  # Returns the custom_add function
    """

    def __init__(self, device_name: str = "device"):
        """
        Create a new registry.
        
        Args:
            device_name: Name of the device (e.g., "Spyre", "CPU")
        """
        logger.info(f"[FLOW] Creating OpRegistry for device: {device_name}")
        self._device_name = device_name
        self._fns: dict[str, Callable] = {}  # Dictionary: op_name -> function

    def register(self, op_name: str):
        """
        Decorator to register a custom operation implementation.
        
        Args:
            op_name: Name of the operation (e.g., "add", "mul")
        
        Returns:
            A decorator function
        """
        logger.debug(f"[FLOW] Preparing to register op: {op_name}")
        def _decorator(fn: Callable) -> Callable:
            self._fns[op_name] = fn  # Store the function in our dictionary
            logger.info(f"[FLOW] ✓ Registered custom operation: '{op_name}' for {self._device_name}")
            return fn
        return _decorator

    def lookup(self, op_name: str) -> Optional[Callable]:
        """
        Look up a registered operation by name.
        
        Args:
            op_name: Name of the operation
        
        Returns:
            The registered function, or None if not found
        """
        result = self._fns.get(op_name)
        if result:
            logger.debug(f"[FLOW] Lookup '{op_name}': Found custom implementation")
        else:
            logger.debug(f"[FLOW] Lookup '{op_name}': Not found (will use generic)")
        return result

    def __contains__(self, op_name: str) -> bool:
        """Check if an operation is registered (allows 'in' operator)."""
        return op_name in self._fns

    @property
    def override_ops(self) -> frozenset:
        """Get all registered operation names."""
        ops = frozenset(self._fns.keys())
        logger.debug(f"[FLOW] Total registered ops: {len(ops)}")
        return ops


class DTypeRegistry:
    """
    A simple registry that maps device format strings to PyTorch data types.
    
    Different devices use different names for data types. This registry
    translates device-specific names to PyTorch's torch.dtype.
    
    Example:
        registry = DTypeRegistry("Spyre")
        registry.register("FP32", torch.float32)
        registry.register("INT32", torch.int32)
        
        # Later, convert device format to PyTorch dtype:
        dtype = registry.resolve("FP32")  # Returns torch.float32
    """

    def __init__(self, device_name: str = "device"):
        """
        Create a new dtype registry.
        
        Args:
            device_name: Name of the device (e.g., "Spyre", "CPU")
        """
        logger.info(f"[FLOW] Creating DTypeRegistry for device: {device_name}")
        self._device_name = device_name
        self._map: dict[str, torch.dtype] = {}  # Dictionary: format_string -> torch.dtype

    def register(self, format_str: str, dtype: torch.dtype) -> None:
        """
        Register a single format string -> dtype mapping.
        
        Args:
            format_str: Device-specific format name (e.g., "FP32")
            dtype: PyTorch data type (e.g., torch.float32)
        """
        logger.debug(f"[FLOW] Registering dtype: {format_str} -> {dtype}")
        self._map[format_str] = dtype

    def register_many(self, mapping: dict[str, torch.dtype]) -> None:
        """
        Register multiple format strings at once.
        
        Args:
            mapping: Dictionary of format_string -> torch.dtype
        
        Example:
            registry.register_many({
                "FP32": torch.float32,
                "INT32": torch.int32,
            })
        """
        logger.info(f"[FLOW] Registering {len(mapping)} dtype mappings for {self._device_name}")
        self._map.update(mapping)
        logger.debug(f"[FLOW] Registered formats: {list(mapping.keys())}")

    def resolve(self, format_str: str) -> torch.dtype:
        """
        Convert a device format string to PyTorch dtype.
        
        Args:
            format_str: Device-specific format name
        
        Returns:
            The corresponding torch.dtype
        
        Raises:
            ValueError: If the format string is not registered
        """
        logger.debug(f"[FLOW] Resolving dtype format: {format_str}")
        if format_str not in self._map:
            logger.error(f"[FLOW] Unknown format string: {format_str}")
            raise ValueError(
                f"Unknown format string '{format_str}' for {self._device_name}. "
                f"Available formats: {list(self._map.keys())}"
            )
        dtype = self._map[format_str]
        logger.debug(f"[FLOW] Resolved {format_str} -> {dtype}")
        return dtype

    def __contains__(self, format_str: str) -> bool:
        """Check if a format string is registered (allows 'in' operator)."""
        return format_str in self._map