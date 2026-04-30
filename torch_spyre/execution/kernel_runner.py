# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("kernel_runner")

# Check if mock device is enabled
MOCK_DEVICE_ENABLED = os.environ.get('TORCH_SPYRE_MOCK_DEVICE', '0') == '1'

# Import launch_kernel conditionally
if MOCK_DEVICE_ENABLED:
    def launch_kernel(graph_path, args):
        """Mock launch_kernel that captures SDSC JSON"""
        print(f"[MOCK_DEVICE] Would launch kernel: {graph_path}")
        try:
            # Try to read and display the SDSC JSON if it exists
            if os.path.exists(graph_path):
                with open(graph_path, 'rb') as f:
                    # CBOR files need special handling, for now just note the path
                    print(f"[MOCK_DEVICE] SDSC graph file: {graph_path}")
            
            # Log the operation
            try:
                from torch_spyre.mock_device_ops import log_operation
                log_operation("launch_kernel", args, None, {"graph": graph_path})
            except ImportError:
                pass
        except Exception as e:
            print(f"[MOCK_DEVICE] Error in mock launch_kernel: {e}")
else:
    from torch_spyre._C import launch_kernel


class SpyreUnimplementedRunner:
    def __init__(self, name: str, op: str):
        self.kernel_name = name
        self.op = op

    def run(self, *args, **kw_args):
        raise RuntimeError(
            f"Invoked {self.kernel_name} which contains unimplemented operation {self.op}"
        )


class SpyreSDSCKernelRunner:
    def __init__(self, name: str, code_dirs: list[str], arg_mappings: list[list[int]]):
        self.kernel_name = name
        self.code_dirs = code_dirs
        self.arg_mappings = arg_mappings

    def run(self, *args, **kw_args):
        if MOCK_DEVICE_ENABLED:
            print(f"[MOCK_DEVICE] Running kernel: {self.kernel_name}")
            print(f"[MOCK_DEVICE] Code directories: {self.code_dirs}")
            print(f"[MOCK_DEVICE] Arg mappings: {self.arg_mappings}")
        
        for i in range(len(self.code_dirs)):
            g2 = os.path.join(self.code_dirs[i], "g2.graph.cbor")
            logger.info(f"RUN: {self.kernel_name}_{i} {g2}")
            actuals = [args[i] for i in self.arg_mappings[i]]
            
            if MOCK_DEVICE_ENABLED:
                # In mock mode, also try to find and display JSON representation
                json_path = os.path.join(self.code_dirs[i], "g2.graph.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            sdsc_json = json.load(f)
                            print("\n" + "="*60)
                            print(f"SDSC JSON for {self.kernel_name}_{i}")
                            print("="*60)
                            print(json.dumps(sdsc_json, indent=2))
                            print("="*60 + "\n")
                    except Exception as e:
                        print(f"[MOCK_DEVICE] Could not read JSON: {e}")
            
            launch_kernel(g2, actuals)
