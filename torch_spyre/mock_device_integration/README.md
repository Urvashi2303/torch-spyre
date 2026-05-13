# Torch-Spyre Mock Device Integration
Enable torch-spyre to work **without physical Spyre hardware** by using CPU-based mock execution. Perfect for development, testing and CI/CD pipelines.

## Setup
### Build and Install
```bash
# Navigate to torch-spyre directory
cd /path/to/torch-spyre

# Set mock-device directory path, mock-device github: https://github.ibm.com/Urvashi-Klair/mock-device/
export MOCKDEVICE_DIR=/path/to/mock-device

# Build and install (mock-device + torch-spyre)
./build_wheel_with_stubs.sh
```

That's it! The build script automatically:
1. **Uninstalls** old torch-spyre and mockdevice packages
2. **Rebuilds** mock-device wheel (always fresh, includes latest operations)
3. **Installs** mock-device package
4. **Builds** torch-spyre wheel (no C++ compilation)
5. **Installs** torch-spyre package

### Enable Mock Device
```bash
# Set environment variable before running your code
export TORCH_SPYRE_MOCK_DEVICE=1

# Run your code
python your_script.py
```


## What Happens Behind the Scenes?
When `TORCH_SPYRE_MOCK_DEVICE=1` is set:

1. **Tensor Operations** → MockSpyreTensor intercepts `.to("spyre")` calls
2. **Compilation** → PyTorch Inductor generates SDSC JSON artifacts
3. **Execution** → MockDevice validates SDSC and runs operations on CPU
4. **Results** → Returned from mock-device instead of Spyre hardware


## Files in This Directory
### `_C_stub.py`
- **Purpose**: Replaces C++ extension module `torch_spyre._C`
- **When Used**: Automatically when `USE_STUBS=1` during build
- **What It Provides**: Python implementations of SpyreTensorLayout, DataFormats, etc.

### `mock_spyre_tensor.py`
- **Purpose**: Creates fake "spyre" device tensors (actually on CPU)
- **When Used**: When `TORCH_SPYRE_MOCK_DEVICE=1` is set
- **What It Does**: Intercepts tensor operations and enables torch.compile() integration

### `mock_device_ops.py`
- **Purpose**: Logs tensor operations for debugging
- **When Used**: Automatically with mock device mode
- **What It Does**: Tracks operations when tensors are moved to "spyre" device

### `mock_compile_patches.py`
- **Purpose**: Patches compilation pipeline for mock device
- **When Used**: During torch.compile() with mock device
- **What It Does**: Enables fallbacks for operations like cat, matmul, full


## Troubleshooting
### "PyTorch is not linked with support for spyre devices"
**Solution**: Set `TORCH_SPYRE_MOCK_DEVICE=1` before running your code.

```bash
export TORCH_SPYRE_MOCK_DEVICE=1
python your_script.py
```

### "No module named 'mockdevice'"
**Solution**: Make sure you set MOCKDEVICE_DIR as mentioned in the setup instructions

### "unknown operation" warnings
**Solution**: The operation may not be registered in mockdevice. Check:
- `mock-device/backends/spyre/ops.py` - Operation implementations
- `mock-device/backends/spyre/validator.py` - KNOWN_OP_FUNCS list
