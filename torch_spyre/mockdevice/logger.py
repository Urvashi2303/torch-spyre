"""Logging utilities for mock device."""

import sys
from typing import Any
from .config import get_config


class MockDeviceLogger:
    """Logger for mock device with configurable verbosity."""
    
    def __init__(self, component: str = ""):
        self.component = component
    
    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on debug level."""
        config = get_config()
        if not config.enabled:
            return False
        
        debug_level = config.debug_level
        if debug_level == "quiet":
            return False
        elif debug_level == "normal":
            return level in ("info", "warning", "error")
        else:  # verbose
            return True
    
    def _format_message(self, level: str, message: str) -> str:
        """Format log message with prefix and component."""
        config = get_config()
        prefix = config.log_prefix
        component_str = f"[{self.component}]" if self.component else ""
        return f"{prefix}{component_str}[{level.upper()}] {message}"
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        if self._should_log("debug"):
            formatted = self._format_message("debug", message)
            if kwargs:
                formatted += f" | {kwargs}"
            print(formatted, file=sys.stderr)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        if self._should_log("info"):
            formatted = self._format_message("info", message)
            if kwargs:
                formatted += f" | {kwargs}"
            print(formatted, file=sys.stderr)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        if self._should_log("warning"):
            formatted = self._format_message("warning", message)
            if kwargs:
                formatted += f" | {kwargs}"
            print(formatted, file=sys.stderr)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        if self._should_log("error"):
            formatted = self._format_message("error", message)
            if kwargs:
                formatted += f" | {kwargs}"
            print(formatted, file=sys.stderr)
    
    def stage(self, stage_name: str, details: str = "") -> None:
        """Log execution stage transition."""
        if self._should_log("info"):
            separator = "=" * 60
            message = f"\n{separator}\n>>> STAGE: {stage_name}"
            if details:
                message += f"\n>>> {details}"
            message += f"\n{separator}"
            print(message, file=sys.stderr)


def get_logger(component: str = "") -> MockDeviceLogger:
    """Get a logger for a specific component."""
    return MockDeviceLogger(component)
