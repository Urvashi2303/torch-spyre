#!/bin/bash
# Copyright 2025 The Torch-Spyre Authors.
# Build script for torch-spyre with stub dependencies (Python-only, no C++ compilation)

set -e

echo "=========================================="
echo "Building torch-spyre wheel with stubs"
echo "Python-only build (no C++ compilation)"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Define mockdevice path
MOCKDEVICE_DIR="../mock-device"
MOCKDEVICE_WHEEL="$MOCKDEVICE_DIR/dist/mockdevice-0.1.0-py3-none-any.whl"

# Step 1: Build mockdevice wheel if it doesn't exist
echo ""
echo "Step 1: Building mockdevice wheel..."
echo "=========================================="
if [ ! -f "$MOCKDEVICE_WHEEL" ]; then
    echo "mockdevice wheel not found, building it..."
    cd "$MOCKDEVICE_DIR"
    ./build_wheel.sh
    cd "$SCRIPT_DIR"
else
    echo "mockdevice wheel already exists: $MOCKDEVICE_WHEEL"
fi

# Step 2: Install mockdevice wheel
echo ""
echo "Step 2: Installing mockdevice wheel..."
echo "=========================================="
pip install --force-reinstall "$MOCKDEVICE_WHEEL"

# Step 3: Build torch-spyre wheel
echo ""
echo "Step 3: Building torch-spyre wheel..."
echo "=========================================="

# Enable stub mode - this skips C++ compilation
export USE_STUBS=1

echo "Environment setup:"
echo "  USE_STUBS=$USE_STUBS (C++ compilation disabled)"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.egg-info
python3 setup.py clean 2>/dev/null || true

# Build the wheel using setup.py with USE_STUBS=1
echo ""
echo "Building torch-spyre wheel with USE_STUBS=1..."
python3 setup.py bdist_wheel

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "Wheel file(s) created in dist/ directory:"
ls -lh dist/*.whl 2>/dev/null || echo "No wheel files found"
echo ""
echo "To install the wheel:"
echo "  pip install dist/torch_spyre-*.whl"
echo ""
echo "Note: This is a Python-only build using mock device."
echo "No C++ extensions are compiled. The mock device enables:"
echo "  - PyTorch Inductor integration"
echo "  - SDSC JSON generation"
echo "  - CPU-based execution for testing"

# Made with Bob
