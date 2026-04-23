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
echo "Building wheel with USE_STUBS=1..."
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
