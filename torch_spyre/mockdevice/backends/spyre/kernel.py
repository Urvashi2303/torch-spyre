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
        for i in range(len(self.code_dirs)):
            code_dir: str = self.code_dirs[i]
            arg_mapping = self.arg_mappings[i]
            
            input_tensors = {}
            for j, idx in enumerate(arg_mapping):
                tensor_name = f"Tensor{j}"
                input_tensors[tensor_name] = args[idx].to("cpu")
            
            # Load mock_op_specs.json (OpSpec format only)
            mock_specs_json = os.path.join(code_dir, "mock_op_specs.json")
            
            if not os.path.exists(mock_specs_json):
                logger.error(f"[FLOW] mock_op_specs.json not found in {code_dir}")
                raise FileNotFoundError(f"mock_op_specs.json not found in {code_dir}")
            
            outputs, used_inputs, graph = self.device.submit(
                mock_specs_json,
                input_tensors,
            )
        
        self.device.synchronize()
        self.device.shutdown()
        
        logger.stage("KERNEL_COMPLETE", f"Kernel '{self.kernel_name}' complete")


