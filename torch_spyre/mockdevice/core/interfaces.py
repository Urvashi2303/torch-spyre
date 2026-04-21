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

    def _generic_cpu_dispatch(self, op_name: str, inputs: list[torch.Tensor],
                              attributes: dict) -> list[torch.Tensor]:
        """
        Execute any torch.ops.aten operation by name with attribute mapping.
        
        Steps:
        1. Resolve torch.ops.aten.<op_name>
        2. Apply attr_map key translation, because Spyre device may use different naming conventions than PyTorch
        3. Skip internal keys (prefixed with "_")
        4. Upcast inputs to precision_dtype if specified, to convert all input tensors to higher precision for testing accuracy
        5. Normalize tuple returns to list
        
        Args:
            op_name: Name of the ATen operation (e.g., "add", "matmul")
            inputs: List of input tensors
            attributes: Dictionary of operation attributes
        
        Returns:
            List of output tensors
        
        Raises:
            AttributeError: If operation doesn't exist in torch.ops.aten
        """
        print(f"   _generic_cpu_dispatch() called")
        print(f"     Operation: torch.ops.aten.{op_name}")
        print(f"     Inputs: {len(inputs)} tensors")
        print(f"     Attributes: {attributes}")
        
        # Step 1: Resolve the ATen operation
        print(f"   Step 1: Resolving torch.ops.aten.{op_name}...")
        try:
            aten_op = getattr(torch.ops.aten, op_name)
            print(f"      Operation found:", aten_op)
        except AttributeError:
            print(f"      Operation not found")
            logger.error(f"Operation 'torch.ops.aten.{op_name}' not found")
            raise
        
        # Step 2 & 3: Translate attributes and skip internal keys
        # attr_map translates SDSC attribute names to PyTorch parameter names.
        # Keys starting with "_" are skipped (internal framework use).
        # Keys with trailing "_" are either translated via attr_map or have the trailing "_" stripped.
        # Example: attr_map = {"fidelity_": "approximate"} translates fidelity_ -> approximate
        #          If not in attr_map, "axis_" -> "axis" (trailing underscore stripped)
        print(f"   Step 2: Filtering attributes...")
        translated_attrs = {}
        skipped_keys = []
        for key, value in attributes.items():
            # Skip internal keys (start with "_")
            if key.startswith("_"):
                skipped_keys.append(key)
                continue
            
            # Translate key using attr_map, or strip trailing underscore if not in map
            new_key = self._attr_map.get(key, key.rstrip("_"))
            translated_attrs[new_key] = value
        
        if skipped_keys:
            print(f"     Skipped internal keys: {skipped_keys}")
        print(f"     Final attributes: {translated_attrs if translated_attrs else '(none)'}")
        
        # Step 4: Upcast inputs to precision dtype if specified
        if self._precision_dtype is not None:
            print(f"   Step 3: Upcasting to {self._precision_dtype}...")
            inputs = [
                inp.to(self._precision_dtype) if isinstance(inp, torch.Tensor) else inp
                for inp in inputs
            ]
        
        # Execute the operation
        # Attempt dispatch with attrs, fall back to no-attrs for ops that
        # don't accept kwargs (e.g., relu, tanh, add without alpha parameter)
        print(f"   Step 4: Executing torch.ops.aten.{op_name}()")
        print(f"     Input shapes: {[tuple(inp.shape) for inp in inputs]}")
        
        try:
            result = aten_op(*inputs, **translated_attrs)
        except (TypeError, RuntimeError):
            # Fall back to calling without attributes if the op doesn't accept them
            result = aten_op(*inputs)
        
        print(f"      Execution successful")
        print(f"     Output shape: {tuple(result.shape) if hasattr(result, 'shape') else 'N/A'}")
        
        # Step 5: Normalize tuple returns to list
        if isinstance(result, tuple):
            return list(result)
        return [result] if not isinstance(result, list) else result

    def execute(self, spec: MockOpSpec, input_tensors: dict) -> dict:
        """
        Execute the op spec with three-tier dispatch strategy.
        
        Tier 1: Try custom registry implementation
        Tier 2: Fall back to generic ATen operation
        Tier 3: Raise NotImplementedError if neither works
        
        Buffer management:
        - INPUT tensors (including dual-role INPUT+OUTPUT) are bound first
        - For in-place tensors: bound as input, overwritten after dispatch
        - Output tensors are collected from buffer_map by checking "OUTPUT" in TensorDescriptor.roles
        
        Execution order:
        - Ops are executed in the order they appear in compute_ops from the compiled artifacts
        
        Args:
            spec: The MockOpSpec to execute
            input_tensors: Dictionary mapping tensor names to PyTorch tensors
        
        Returns:
            Dictionary of output tensors (only those with OUTPUT role)
        
        Raises:
            RuntimeError: If tensor not found in buffer map or output count mismatch
            NotImplementedError: If operation not found in registry or ATen
        """
        logger.stage("DISPATCH", f"Executing MockOpSpec '{spec.op_spec_name}'")
        
        # Initialize buffer map with all tensors
        buffer_map: dict[str, torch.Tensor] = {}
        
        # Step 1: Bind input tensors (apply layout transformation)
        logger.info(f"[FLOW] Binding {len(input_tensors)} input tensors")
        for name, tensor in input_tensors.items():
            if name not in spec.tensors:
                logger.warning(f"Input tensor '{name}' not found in spec, skipping")
                continue
            
            td = spec.tensors[name]
            
            # Convert from hardware layout to contiguous
            contiguous_tensor = self._layout.to_contiguous(tensor, td)
            buffer_map[name] = contiguous_tensor
            
            if self.verbose:
                logger.debug(f"Bound input tensor '{name}': shape={list(contiguous_tensor.shape)}")
        
        # Step 2: Execute each compute operation in order
        logger.info(f"[FLOW] Executing {len(spec.compute_ops)} compute operations")
        
        for i, compute_op in enumerate(spec.compute_ops):
            if self.verbose:
                logger.debug(f"[FLOW] ComputeOp {i}: {compute_op.op_func_name}")
            
            # Gather input tensors for this operation
            try:
                inputs = [buffer_map[name] for name in compute_op.input_names]
                print(f"DEBUG: Gathered {len(inputs)} inputs for op '{compute_op.op_func_name}': {compute_op.input_names}")
                print(f"DEBUG: Buffer map keys: {list(buffer_map.keys())}")
            except KeyError as e:
                logger.error(f"Missing input tensor for operation '{compute_op.op_func_name}': {e}")
                raise RuntimeError(f"Tensor {e} not found in buffer map") from e
            
            # THREE-TIER DISPATCH
            print(f"\n{'='*60}")
            print(f"THREE-TIER DISPATCH for operation: '{compute_op.op_func_name}'")
            print(f"{'='*60}")
            
            # TIER 1: Try custom registry implementation
            print(f"[TIER 1] Checking custom registry for operation '{compute_op.op_func_name}'")
            custom_impl = self._registry.lookup(compute_op.op_func_name)
            
            if custom_impl is not None:
                # Use custom implementation
                print(f"[TIER 1]  Found custom implementation for '{compute_op.op_func_name}'")
                print(f"[TIER 1] Executing custom implementation...")
                
                try:
                    outputs = custom_impl(inputs, compute_op, self.verbose)
                    print(f"[TIER 1]  Custom implementation succeeded")
                except Exception as e:
                    logger.error(f"Custom implementation failed for '{compute_op.op_func_name}': {e}")
                    raise
            
            else:
                # TIER 2: Try generic ATen operation
                print(f"[TIER 1]  Not found in custom registry")
                print(f"[TIER 2] Trying generic ATen fallback: torch.ops.aten.{compute_op.op_func_name}")
                
                try:
                    outputs = self._generic_cpu_dispatch(
                        compute_op.op_func_name,
                        inputs,
                        compute_op.attributes
                    )
                    print(f"[TIER 2]  Generic ATen operation succeeded")
                except AttributeError as e:
                    # TIER 3: Operation doesn't exist anywhere
                    print(f"[TIER 2]  Operation not found in torch.ops.aten")
                    print(f"[TIER 3]  FAILED - Operation '{compute_op.op_func_name}' not implemented anywhere")
                    logger.error(f"Operation '{compute_op.op_func_name}' not found in registry or torch.ops.aten")
                    raise NotImplementedError(
                        f"Operation '{compute_op.op_func_name}' is not implemented. "
                        f"Not found in custom registry '{self._registry._device_name}' or torch.ops.aten"
                    ) from e
            
            print(f"{'='*60}\n")
            
            # Validate output count
            if len(outputs) != len(compute_op.output_names):
                logger.error(
                    f"Output count mismatch for '{compute_op.op_func_name}': "
                    f"expected {len(compute_op.output_names)}, got {len(outputs)}"
                )
                raise RuntimeError(
                    f"Operation '{compute_op.op_func_name}' returned {len(outputs)} outputs, "
                    f"but spec declares {len(compute_op.output_names)} outputs"
                )
            
            # Store outputs in buffer map (apply layout transformation)
            for output_tensor, output_name in zip(outputs, compute_op.output_names):
                td = spec.tensors[output_name]
                
                # Convert from contiguous to hardware layout
                hardware_tensor = self._layout.from_contiguous(output_tensor, td)
                buffer_map[output_name] = hardware_tensor
                
                if self.verbose:
                    logger.debug(f"Stored output tensor '{output_name}': shape={list(hardware_tensor.shape)}")
        
        # Step 3: Extract and return only OUTPUT tensors
        output_tensors = {
            name: buffer_map[name]
            for name, td in spec.tensors.items()
            if td.is_output()
        }
        
        logger.info(f"[FLOW] Execution complete, returning {len(output_tensors)} output tensors")
        
        return output_tensors


# ---------------------------------------------------------------------------
# Mock device
# ---------------------------------------------------------------------------

class AbstractMockDevice(ABC):
    """
    MockSpyreDevice: Top-level device class. Chains Parser  Validator  Dispatcher
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

        spec = self._load_artifact(artifact)
        # print(f"spec: {spec}")
        
        self._on_submit_start(spec, artifact, **kwargs)
        print("validate spec")
        self._validator.validate(spec)

        if input_tensors is None:
            logger.error(f"[FLOW] No input tensors provided for '{spec.op_spec_name}'")
            raise ValueError(
                f"Input tensors are required. Please provide input_tensors dictionary "
                f"with keys: {list(spec.input_tensors.keys())}"
            )

        outputs = self._dispatcher.execute(spec, input_tensors)

        logger.stage("COMPLETE", f"{self.device_name} execution complete")
        return outputs, input_tensors, spec
