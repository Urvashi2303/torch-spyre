"""Configuration for mock device behavior"""

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class MockDeviceConfig:
    """Configuration for mock device behavior.
    
    All validations (structural, semantic, and correctness) are always enabled
    to ensure MockOpSpec correctness and prevent runtime errors.
    
    Attributes:
        enabled: Whether mock device is enabled
        debug_level: Level of debug output (quiet, normal, verbose)
        log_prefix: Prefix for all debug log messages
    """
    enabled: bool = False
    debug_level: Literal["quiet", "normal", "verbose"] = "normal"
    log_prefix: str = "[MOCK_DEVICE]"
    
    @classmethod
    def from_env(cls) -> "MockDeviceConfig":
        """Create configuration from environment variables.
        
        Environment variables:
            TORCH_SPYRE_MOCK_DEVICE: Enable mock device (1/true/yes)
            TORCH_SPYRE_MOCK_DEBUG: Debug level (quiet/normal/verbose)
        """
        enabled = _parse_bool(os.getenv("TORCH_SPYRE_MOCK_DEVICE", "0"))
        debug_level = os.getenv("TORCH_SPYRE_MOCK_DEBUG", "normal")
        if debug_level not in ("quiet", "normal", "verbose"):
            debug_level = "normal"
        
        return cls(
            enabled=enabled,
            debug_level=debug_level
        )


def _parse_bool(value: str) -> bool:
    """Parse boolean from string."""
    return value.lower() in ("1", "true", "yes", "on")


# Global configuration instance
_config: MockDeviceConfig | None = None

def set_config(config: MockDeviceConfig) -> None:
    """Set the global mock device configuration."""
    global _config
    _config = config

def get_config() -> MockDeviceConfig:
    """Get the global mock device configuration."""
    global _config
    if _config is None:
        _config = MockDeviceConfig.from_env()
    return _config

def is_mock_enabled() -> bool:
    """Check if mock device is enabled."""
    return get_config().enabled




