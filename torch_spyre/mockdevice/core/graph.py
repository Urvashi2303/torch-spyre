"""
torch_spyre.mockdevice.core.graph
----------------------------------
Generic op-graph data model matching the RFC specification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch
from ..logger import get_logger

logger = get_logger("Graph")

INPUT  = "INPUT"
OUTPUT = "OUTPUT"
ROLES  = frozenset({INPUT, OUTPUT})

@dataclass
class TensorDescriptor:
    """
    Dataclass describing a single tensor
    """
    name:        str                # the tensor identifier used to reference this tensor elsewhere in the graph.
    roles:       set[str]           # a set of role labels such as {"INPUT"}, {"OUTPUT"}, or both. This is later used by is_input(), is_output(), and is_inplace().
    shape:       list[int]          # the logical tensor dimensions, similar to PyTorch tensor sizes.
    stride:      list[int]          # the logical memory stride for each dimension, describing how elements are laid out in memory.
    dtype:       torch.dtype        # the PyTorch data type, such as torch.float16.
    device_size: list[int]          # device-specific layout dimensions. This may differ from logical shape because hardware may store tensors in blocked or transformed layouts.
    dim_map:     list[int]          # maps logical dimensions to device dimensions. This helps interpret how the tensor’s shape is rearranged for the device.
    device_dtype: str               # the device/backend-specific type string, separate from PyTorch’s dtype.
    
    metadata:    dict[str, Any] = field(default_factory=dict)  # an extensible dictionary for extra per-tensor information that does not belong in the core schema.
    
    # the framework view        : shape, stride, dtype
    # the hardware/backend view : device_size, dim_map, device_dtype

    def __post_init__(self):
        logger.debug(f"[FLOW] Created TensorDescriptor: {self.name}, roles={self.roles}, shape={self.shape}")

    def is_input(self)  -> bool: 
        return INPUT in self.roles
    
    def is_output(self) -> bool: 
        return OUTPUT in self.roles
    
    def is_inplace(self) -> bool: 
        return self.roles == ROLES

@dataclass
class ComputeOp:
    """
    Dataclass describing a single compute operation
    """
    op_func_name:        str                    # op name e.g., "matmul", "add", "relu"
    ex_unit:             str                    # the hardware component (e.g., "tensor_core", "vector_unit") that will execute this operation
    input_tensor_names:  list[str]              # List of tensor identifiers that serve as inputs to this operation, references tensors defined as TensorDescriptor in the graph
    output_tensor_names: list[str]              # List of tensor identifiers produced by this operation, references the output tensors that this operation generates
    num_cores:           int                    # number of cores that will be used to execute this operation, imp for parallelization and resource allocation on the hardware

    attributes:          dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        logger.debug(f"[FLOW] Created ComputeOp: {self.op_func_name}, ex_unit={self.ex_unit}, cores={self.num_cores}")

    def resolve_input_name(self, ref: str) -> str:
        return ref.split("-")[0]

    def resolve_output_name(self, ref: str) -> str:
        return ref.split("-")[0]

    @property
    def input_names(self) -> list[str]:
        return [self.resolve_input_name(r) for r in self.input_tensor_names]

    @property
    def output_names(self) -> list[str]:
        return [self.resolve_output_name(r) for r in self.output_tensor_names]

@dataclass
class MockOpSpec:
    """
    Internal DAG representation: op name, dimension map, tensor map, compute ops, core count.
    
    This is the main data structure
    """
    op_spec_name: str                                    # operation name
    dimensions:   dict[str, int]                         # mapping of symbolic dimension names to their concrete sizes. This allows operations to reference dimensions by name (e.g., {"M": 128, "N": 256, "K": 64} for a matrix multiplication)
    tensors:      dict[str, TensorDescriptor]            # dictionary mapping tensor names to their full descriptors
    compute_ops:  list[ComputeOp]                        # an ordered list of compute operations that make up this op spec
    
    metadata:     dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        logger.info(f"[FLOW] Created MockOpSpec: {self.op_spec_name}, tensors={len(self.tensors)}, ops={len(self.compute_ops)}")

    @property
    def input_tensors(self) -> dict[str, TensorDescriptor]:
        return {n: td for n, td in self.tensors.items() if td.is_input()}

    @property
    def output_tensors(self) -> dict[str, TensorDescriptor]:
        return {n: td for n, td in self.tensors.items() if td.is_output()}