"""
torch_spyre.mockdevice.core.interfaces
--------------------------------------
Abstract base classes
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from torch_spyre.mockdevice.core.graph import MockOpSpec
from typing import Any, Optional
import torch
from .graph import MockOpSpec, TensorDescriptor
from ..logger import get_logger

logger = get_logger("Interfaces")

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class AbstractGraphValidator(ABC):
    """
    Artifact Validator: Validates MockOpSpec for structural and semantic correctness.
    
    Independent of runtime interaction.
    """

    @abstractmethod
    def validate(self, spec: MockOpSpec) -> None:
        """Raise `ValueError` if the spec is invalid."""


class BaseGraphValidator(AbstractGraphValidator):
    """
    Concrete base that runs generic structural checks.
    
    Validation sequences four sub-checks in a fixed order:
    - _check_tensors(): verifies every TensorDescriptor
    - _check_compute_ops(): verifies every ComputeOp
    - _check_io_wiring(): resolves tensor references
    - validate_device_specifics(): backend-specific findings
    """

    VALID_ROLES = frozenset({"INPUT", "OUTPUT"})

    def validate(self, spec: MockOpSpec) -> None:
        """
        Validate MockOpSpec for structural correctness.
        
        Runs structural validation checks then device-specific validation.
        Collects all errors and reports them together.
        
        Raises:
            ValueError: If any validation check fails
        """
        logger.stage("VALIDATE", f"Validating MockOpSpec '{spec.op_spec_name}'")
        errors: list[str] = []
        
        self._check_tensors(spec, errors)
        self._check_compute_ops(spec, errors)
        self._check_io_wiring(spec, errors)
        self.validate_device_specifics(spec, errors)
        
        if errors:
            msg = f"MockOpSpec validation failed with {len(errors)} error(s):\n"
            for i, e in enumerate(errors, 1):
                msg += f"  [{i}] {e}\n"
            logger.error(msg.strip())
            raise ValueError(msg.strip())
        
        logger.info(f"[FLOW] MockOpSpec '{spec.op_spec_name}' passed validation")

    def _check_tensors(self, spec: MockOpSpec, errors: list) -> None:
        """Check tensor roles and shapes."""
        if not spec.tensors:
            errors.append("No tensors found in MockOpSpec")
            return
        
        for name, td in spec.tensors.items():
            if not td.roles:
                errors.append(f"Tensor '{name}': roles is empty")
            else:
                unknown = td.roles - self.VALID_ROLES
                if unknown:
                    errors.append(
                        f"Tensor '{name}': unknown role(s) {unknown}. "
                        f"Each role must be 'INPUT' or 'OUTPUT'."
                    )
            
            if not td.shape:
                errors.append(f"Tensor '{name}': shape is empty")
            else:
                if any(s <= 0 for s in td.shape):
                    errors.append(
                        f"Tensor '{name}': all shape dims must be > 0, got {td.shape}"
                    )

    def _check_compute_ops(self, spec: MockOpSpec, errors: list) -> None:
        """Check compute operations have required fields."""
        if not spec.compute_ops:
            errors.append("No compute operations defined")
            return
        
        for i, op in enumerate(spec.compute_ops):
            if not op.op_func_name:
                errors.append(f"compute_ops[{i}]: op_func_name is empty")
            if not op.input_tensor_names:
                errors.append(f"compute_ops[{i}] '{op.op_func_name}': no input tensors")
            if not op.output_tensor_names:
                errors.append(f"compute_ops[{i}] '{op.op_func_name}': no output tensors")

    def _check_io_wiring(self, spec: MockOpSpec, errors: list) -> None:
        """Check tensor references exist and at least one INPUT/OUTPUT present."""
        known = set(spec.tensors)
        
        for i, op in enumerate(spec.compute_ops):
            for ref in op.input_tensor_names + op.output_tensor_names:
                tname = ref.split("-")[0]
                if tname not in known:
                    errors.append(
                        f"compute_ops[{i}] '{op.op_func_name}': ref '{ref}' "
                        f"resolves to '{tname}' which is not in tensors. "
                        f"Known: {sorted(known)}"
                    )
        
        all_roles: set[str] = set()
        for td in spec.tensors.values():
            all_roles |= td.roles
        
        if "INPUT" not in all_roles:
            errors.append("No INPUT tensor found in MockOpSpec")
        if "OUTPUT" not in all_roles:
            errors.append("No OUTPUT tensor found in MockOpSpec")

    def validate_device_specifics(self, spec: MockOpSpec, errors: list) -> None:
        """Override to add backend-specific validation rules."""
        pass


# ---------------------------------------------------------------------------
# Layout transformer
# ---------------------------------------------------------------------------

class AbstractLayoutTransformer(ABC):
    """
    Layout Transformer: Converts between Spyre stick layout and contiguous PyTorch tensors.
    
    If a physical Spyre card is available, it moves the Spyre tensor to and from host.
    """

    @abstractmethod
    def to_contiguous(self, tensor: torch.Tensor, td: TensorDescriptor) -> torch.Tensor:
        """
        Device layout -> contiguous PyTorch tensor.
        
        Performs strip on the input path: if the incoming tensor's shape already
        matches the declared TensorDescriptor.shape (no padding present), it is
        returned as-is via .contiguous(); otherwise a tuple of slice(0, s) indices
        is applied across every dimension to discard the excess elements before
        handing the tensor to the dispatcher.
        """

    @abstractmethod
    def from_contiguous(self, tensor: torch.Tensor, td: TensorDescriptor) -> torch.Tensor:
        """
        Contiguous PyTorch tensor -> device layout.
        
        Intentionally a no-op on the output path - because there is no physical
        HBM to write to in mock mode, no re-padding or DMA transfer is needed,
        and the op result is returned directly in contiguous form.
        """


# ---------------------------------------------------------------------------
# Op dispatcher
# ---------------------------------------------------------------------------

class MockOpDispatcher:
    """
    MockOpDispatcher: Walks the validated MockOpSpec and dispatches each ComputeOp
    to its corresponding kernels.
    
    Maps Spyre ops to ATen CPU kernels. Calls to_contiguous() on every input tensor
    immediately after binding and from_contiguous() on every output tensor before
    storing it in the buffer map.
    """

    def __init__(self, registry, layout: "AbstractLayoutTransformer", attr_map: dict | None = None,
                 precision_dtype=None, verbose: bool = False):
        self._registry = registry
        self._layout = layout
        self._attr_map = attr_map or {}
        self._precision_dtype = precision_dtype
        self.verbose = verbose

    def execute(self, spec: MockOpSpec, input_tensors: dict) -> dict:
        """
        Execute the op spec and return {name: tensor} for every OUTPUT tensor.
        
        Buffer management:
        - INPUT tensors (including dual-role INPUT+OUTPUT) are bound first
        - For in-place tensors: bound as input, overwritten after dispatch
        - Output tensors are collected from buffer_map by checking "OUTPUT" in TensorDescriptor.ds_type
        
        Execution order:
        - Ops are executed in the order they appear in compute_ops from the compiled artifacts
        """
        logger.stage("DISPATCH", f"Executing MockOpSpec '{spec.op_spec_name}'")
        
        # Execution logic would go here - removed for flow-only version
        return {}


# ---------------------------------------------------------------------------
# Mock device
# ---------------------------------------------------------------------------

class AbstractMockDevice(ABC):
    """
    MockSpyreDevice: Top-level device class. Chains Parser → Validator → Dispatcher
    in a submit() method. Matches real SpyreDevice interface.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._initialized = False
        self._validator = self._make_validator()
        self._dispatcher = self._make_dispatcher()
        logger.info(f"[FLOW] {self.device_name} mock device initialized")

    @property
    @abstractmethod
    def device_name(self) -> str:
        """Human-readable device identifier."""

    @abstractmethod
    def _make_validator(self) -> AbstractGraphValidator:
        """Return the validator for this device's op spec."""

    @abstractmethod
    def _make_dispatcher(self) -> MockOpDispatcher:
        """Return the op dispatcher for this device."""

    @abstractmethod
    def _load_artifact(self, artifact: Any) -> Any:
        """Normalise the caller's artifact argument."""

    def _on_submit_start(self, spec: MockOpSpec, artifact: Any, **kwargs: Any) -> None:
        """
        Called after parsing, before validation.  Override to emit
        device-specific log lines (e.g. the 'Generating …' / 'RUN: …'
        lines that mirror the real Spyre runtime output).
        """
        if self.verbose:
            print(
                f"[{self.device_name}] op='{spec.op_spec_name}' "
                f"tensors={list(spec.tensors)} "
                f"ops={[op.op_func_name for op in spec.compute_ops]}"
            )

    def _generate_random_inputs(self, spec: MockOpSpec) -> dict[str, torch.Tensor]:
        """Generate random tensors for every INPUT tensor."""
        return {}

    def initialize(self) -> None:
        logger.stage("INITIALIZE", f"{self.device_name} device")
        self._initialized = True

    def shutdown(self) -> None:
        logger.stage("SHUTDOWN", f"{self.device_name} device")
        self._initialized = False

    def synchronize(self) -> None:
        """No-op: mock execution is always synchronous on the host."""
        pass

    def submit(self, artifact: Any, input_tensors: Optional[dict[str, torch.Tensor]] = None,
               **kwargs: Any) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], MockOpSpec]:
        """
        Full pipeline: load artifact -> parse -> validate -> dispatch.
        
        The submit() method implements the full validation pipeline in a fixed four-stage sequence:
        1. _load_artifact() normalises the artifact argument into a Python object
        2. Artifact is converted into a device-agnostic MockOpSpec
        3. Validator checks structural correctness and emits errors if any
        4. MockOpDispatcher.execute() runs each ComputeOp against a CPU ATen reference implementation,
           returning (outputs, used_inputs, spec). The three-element return tuple was specifically
           designed for MockKernelRunner: used_inputs prevents a spurious delta when the runner
           generates random inputs internally, and spec lets the runner resolve output tensor names
           by role rather than by hardcoded position.
        """
        logger.stage("SUBMIT", f"{self.device_name} kernel submission")
        
        if not self._initialized:
            logger.error(f"[FLOW] {self.device_name} is not initialized")
            raise RuntimeError(f"{self.device_name} is not initialised. Call initialize() first.")

        raw = self._load_artifact(artifact)

        if isinstance(raw, MockOpSpec):
            spec: MockOpSpec = raw
        else:
            logger.error(f"[FLOW] Unsupported artifact type after load: {type(raw).__name__}")
            raise TypeError("Mock device expects _load_artifact() to return MockOpSpec")

        self._on_submit_start(spec, artifact, **kwargs)
        self._validator.validate(spec)

        if input_tensors is None:
            input_tensors = self._generate_random_inputs(spec)

        outputs = self._dispatcher.execute(spec, input_tensors)

        logger.stage("COMPLETE", f"{self.device_name} execution complete")
        return outputs, input_tensors, spec
