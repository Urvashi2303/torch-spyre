"""
torch_spyre.mockdevice.backends.spyre.kernel
--------------------------------------------
SpyreSDSCMockKernelRunner
"""

import os
from .device import MockSpyreDevice
from ...logger import get_logger

logger = get_logger("SpyreKernel")

class SpyreUnimplementedRunner:
    """Placeholder for unimplemented operations."""
    
    def __init__(self, name: str, op: str):
        logger.warning(f"[FLOW] Created unimplemented runner for kernel '{name}' with op '{op}'")
        self.kernel_name = name
        self.op = op

    def run(self, *args, **kw_args):
        logger.error(f"[FLOW] Attempted to run unimplemented kernel '{self.kernel_name}' with op '{self.op}'")
        raise RuntimeError(f"Invoked {self.kernel_name} which contains unimplemented operation {self.op}")

class SpyreSDSCMockKernelRunner:
    """Mock kernel runner - matches hardware runner interface."""
    
    def __init__(self, name: str, code_dirs: list[str], arg_mappings: list[list[int]]):
        self.kernel_name = name
        self.code_dirs = code_dirs
        self.arg_mappings = arg_mappings
        
        verbose = int(os.getenv("MOCK_SPYRE_VERBOSE", "0"))
        self.device = MockSpyreDevice(verbose=bool(verbose))
        
        logger.info(f"[FLOW] SpyreSDSCMockKernelRunner initialized: {name} with {len(code_dirs)} operations")

    def run(self, *args, **kw_args):
        """Execute the kernel using the mock device - handles multiple operations."""
        logger.stage("KERNEL_RUN", f"Running kernel '{self.kernel_name}' with {len(self.code_dirs)} operations")
        
        self.device.initialize()
        logger.info(f"[FLOW] MockSpyreDevice runner is initialized")
        
        # Process each operation (matching hardware runner behavior)
        final_outputs = None
        for i in range(len(self.code_dirs)):
            code_dir: str = self.code_dirs[i]
            arg_mapping = self.arg_mappings[i]
            
            input_tensors = {}
            for j, idx in enumerate(arg_mapping):
                tensor_name = f"Tensor{j}"
                input_tensors[tensor_name] = args[idx].to("cpu")
            
            # Pass code_dir directly - device will look for mock_op_specs.json
            outputs, used_inputs, graph = self.device.submit(
                code_dir,
                input_tensors,
            )
            
            # Update args in-place with outputs to enable operation chaining
            # This allows subsequent operations to use outputs from previous operations
            output_names = [
                name for name, td in graph.tensors.items()
                if td.is_output()
            ]
            
            # Map output tensors back to their corresponding args indices
            for j, idx in enumerate(arg_mapping):
                tensor_name = f"Tensor{j}"
                if tensor_name in outputs and tensor_name in output_names:
                    # Copy output back to the original args buffer in-place
                    out_tensor = outputs[tensor_name]
                    if args[idx].dtype != out_tensor.dtype:
                        out_tensor = out_tensor.to(args[idx].dtype)
                    args[idx].copy_(out_tensor)
                    logger.debug(f"[FLOW] Updated args[{idx}] with output {tensor_name}")
            
            final_outputs = outputs  # Keep last operation's outputs
        
        self.device.synchronize()
        self.device.shutdown()
        
        logger.stage("KERNEL_COMPLETE", f"Kernel '{self.kernel_name}' complete")
        
        # Return output tensors as a list (matching hardware runner behavior)
        return list(final_outputs.values()) if final_outputs else []
