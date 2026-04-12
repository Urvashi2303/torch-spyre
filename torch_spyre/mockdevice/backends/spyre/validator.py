"""
torch_spyre.mockdevice.backends.spyre.validator
-----------------------------------------------
MockOpSpecValidator - Validates MockOpSpec for Spyre device
"""
from __future__ import annotations
from ...core import BaseGraphValidator, MockOpSpec
from ...logger import get_logger

logger = get_logger("MockOpSpecValidator")

KNOWN_OP_FUNCS = frozenset({
    "maxnonstick", "sumnonstick", "realdiv"
})

KNOWN_EX_UNITS = frozenset({"sfp"})

class MockOpSpecValidator(BaseGraphValidator):
    """
    Validates MockOpSpec for Spyre device.
    
    Extends BaseGraphValidator with Spyre-specific validation rules
    for operations, execution units, and device constraints.
    """

    def validate_device_specifics(self, spec: MockOpSpec, errors: list) -> None:
        logger.info("[FLOW] MockOpSpecValidator.validate_device_specifics() - Running Spyre-specific validation")

