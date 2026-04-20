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

import json
import tempfile
from typing import Any
import os
import subprocess

from torch._inductor.runtime.runtime_utils import cache_dir
from torch_spyre._C import convert_artifacts
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.logging_utils import get_inductor_logger, _get_env_bool
from torch_spyre._inductor.op_spec import OpSpec, UnimplementedOp
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner


def _serialize_specs_bundle(kernel_name: str, specs: list[OpSpec]) -> dict:
    """
    Serialize OpSpec list to mock_op_specs.json format.
    
    Converts inductor OpSpec objects to the JSON format expected by mock device.
    """
    serialized_specs = []
    
    for idx, spec in enumerate(specs):
        # Convert OpSpec to mock device format
        spec_dict = {
            "index": idx,
            "op": spec.op,
            "is_reduction": spec.is_reduction,
            "iteration_space": {
                str(sym): {
                    "range": int(range_val) if hasattr(range_val, '__int__') else range_val,
                    "core_division": core_div
                }
                for sym, (range_val, core_div) in spec.iteration_space.items()
            },
            "args": [
                {
                    "is_input": arg.is_input,
                    "arg_index": arg.arg_index,
                    "device_dtype": str(arg.device_dtype),
                    "device_size": arg.device_size,
                    "device_coordinates": [str(coord) for coord in arg.device_coordinates],
                    "allocation": arg.allocation if arg.allocation else {}
                }
                for arg in spec.args
            ],
            "op_info": spec.op_info if spec.op_info else {}
        }
        serialized_specs.append(spec_dict)
    
    return {
        "kernel_name": kernel_name,
        "num_specs": len(specs),
        "specs": serialized_specs
    }

logger = get_inductor_logger("sdsc_compile")

_SDSC_BUNDLE = _get_env_bool("SPYRE_SUPERDSC_BUNDLE", True)


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    kernel_output_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{kernel_name}_")
    return kernel_output_dir


class SpyreAsyncCompile:
    def __init__(self) -> None:
        pass

    def sdsc(self, kernel_name: str, specs: list[OpSpec | UnimplementedOp]):
        # 1. Generate SDSC.json for each OpSpec
        sdscs_json = []
        arg_mappings = []
        for ks in specs:
            if isinstance(ks, UnimplementedOp):
                print(f"WARNING: Compiling unimplemented {ks.op} to runtime exception")
                return SpyreUnimplementedRunner(kernel_name, ks.op)

            sdsc_json, arg_map = compile_op_spec(kernel_name, ks)
            sdscs_json.append(sdsc_json)
            arg_mappings.append(arg_map)

        # Write SDSCs to file system, invoke backend compiler, and return KernelRunner
        kernel_output_dir = get_output_dir(kernel_name)
        
        # Check if mock mode is enabled via environment variable
        mock_enabled = os.getenv("MOCK_SPYRE", "0") == "1"
        
        if _SDSC_BUNDLE:
            # Generate mock_op_specs.json for mock device
            if mock_enabled:
                specs_path = os.path.join(kernel_output_dir, "mock_op_specs.json")
                logger.info(f"[FLOW] Saving bundled mock specs: {specs_path}")
                serialized_specs = _serialize_specs_bundle(kernel_name, specs)
                with open(specs_path, "w") as file:
                    logger.info(f"Generating {file.name}")
                    json.dump(serialized_specs, file, indent=2)
            
            # Generate SDSC JSON files (for both mock and hardware)
            for idx, sdsc_json in enumerate(sdscs_json):
                with open(
                    os.path.join(kernel_output_dir, f"sdsc_{idx}.json"), "w"
                ) as file:
                    logger.info(f"Generating {file.name}")
                    json.dump(sdsc_json, file, indent=2)
            
            # Generate bundle.mlir (for hardware compilation)
            with open(os.path.join(kernel_output_dir, "bundle.mlir"), "w") as file:
                logger.info(f"Generating {file.name}")
                file.write("module {\n")
                file.write("\tfunc.func @sdsc_bundle() {\n")
                for i in range(len(sdscs_json)):
                    file.write(
                        '\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_'
                        + f"{i}"
                        + '.json"}\n'
                    )
                file.write("\t\treturn\n")
                file.write("\t}\n")
                file.write("}\n")

            if mock_enabled:
                logger.info("[FLOW] MOCK_SPYRE=1 detected - Attaching MockKernelRunner")
                from torch_spyre.mockdevice.backends.spyre.kernel import SpyreSDSCMockKernelRunner
                
                return SpyreSDSCMockKernelRunner(
                    kernel_name,
                    [kernel_output_dir],
                    arg_mappings
                )
            else:
                logger.info("[FLOW] MOCK_SPYRE=0 - Following hardware execution path")
                subprocess.run(
                    ["dxp_standalone", "--bundle", "-d", kernel_output_dir], check=True
                )
                convert_artifacts(kernel_output_dir)

                return SpyreSDSCKernelRunner(kernel_name, [kernel_output_dir], arg_mappings)
        else:
            # Process each SuperDSC separately
            sdsc_dirs = []
            
            # Generate mock_op_specs.json for mock device (non-bundle mode)
            if mock_enabled:
                for idx, spec in enumerate(specs):
                    if isinstance(spec, UnimplementedOp):
                        continue
                    kernel_output_dir = get_output_dir(kernel_name)
                    subdir = os.path.join(kernel_output_dir, "execute", kernel_name)
                    os.makedirs(subdir, exist_ok=True)
                    
                    # Create single-spec bundle for this operation
                    specs_path = os.path.join(kernel_output_dir, "mock_op_specs.json")
                    logger.info(f"[FLOW] Saving mock spec {idx}: {specs_path}")
                    serialized_spec = _serialize_specs_bundle(f"{kernel_name}_{idx}", [spec])
                    with open(specs_path, "w") as file:
                        json.dump(serialized_spec, file, indent=2)
                    
                    sdsc_dirs.append(kernel_output_dir)
            
            # Generate SDSC JSON files (for both mock and hardware)
            for sdsc_json in sdscs_json:
                kernel_output_dir = get_output_dir(kernel_name)
                subdir = os.path.join(kernel_output_dir, "execute", kernel_name)
                os.makedirs(subdir, exist_ok=True)
                with open(os.path.join(subdir, "sdsc.json"), "w") as file:
                    logger.info(f"Generating {file.name}")
                    json.dump(sdsc_json, file, indent=2)
                sdsc_dirs.append(kernel_output_dir)

            if mock_enabled:
                logger.info("[FLOW] MOCK_SPYRE=1 detected - Attaching MockKernelRunner")
                from torch_spyre.mockdevice.backends.spyre.kernel import SpyreSDSCMockKernelRunner
                
                return SpyreSDSCMockKernelRunner(kernel_name, sdsc_dirs, arg_mappings)
            else:
                logger.info("[FLOW] MOCK_SPYRE=0 - Following hardware execution path")
                for dir in sdsc_dirs:
                    subprocess.run(["dxp_standalone", "-d", dir], check=True)
                    convert_artifacts(dir)

                return SpyreSDSCKernelRunner(kernel_name, sdsc_dirs, arg_mappings)

    def wait(self, scope: dict[str, Any]) -> None:
        pass
