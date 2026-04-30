# Torch-Spyre Mock Device Integration

This directory contains **integration files** that enable mock device functionality in torch-spyre when building in stub mode (USE_STUBS=1).

## Files & Purpose

### `_C_stub.py`
- **Purpose**: Replaces `torch_spyre._C` module when building without C++ extensions
- **Used by**: `torch_spyre/_C.py` imports this when USE_STUBS=1
- **What it does**: Provides dummy implementations of C++ functions (DataFormats, SpyreTensorLayout, etc.)

### `mock_device_ops.py`
- **Purpose**: Intercepts tensor operations for SDSC logging
- **Used by**: `torch_spyre/__init__.py` loads this when MOCK_DEVICE=1
- **What it does**: Logs operations when tensors are moved to "spyre" device

### `mock_spyre_tensor.py`
- **Purpose**: Creates fake "spyre" device tensors (actually on CPU)
- **Used by**: `torch_spyre/_monkey_patch.py` and `torch_spyre/__init__.py`
- **What it does**: Tricks torch.compile() into using Spyre inductor backend → generates SDSC JSON

## Core MockDevice Package

The core mockdevice implementation (validation, execution, etc.) is in a **separate package**:
- Git: `https://github.ibm.com/Urvashi-Klair/mock-device`

## Build Process

Update MOCKDEVICE_DIR with path for mock-device directory

The build script automatically handles mockdevice build:
```bash
./build_wheel_with_stubs.sh
```

This:
1. Builds mockdevice wheel
2. Installs mockdevice wheel
3. Builds torch-spyre with USE_STUBS=1 (uses files in this directory)