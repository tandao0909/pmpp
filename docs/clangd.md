In the process of writing CUDA code in VSCode, we want to have some completion.
While we can install ClangD extension, when using the header from pytorch, we would need to include the path to the header.

## Install PyTorch

```bash
# Uv commands
uv init
uv add pytorch

## Finding PyTorch include paths
# Activate your virtual environment first
source .venv/bin/activate

# Find PyTorch installation location
python -c "import torch; print(torch.__file__)"
# This gives you the path to torch/__init__.py

# From the torch path, construct the include paths:
# 1. Replace "__init__.py" with "include"
# 2. For the API include, add "/torch/csrc/api/include" to the base include path

# Example output:
# /home/user/.venv/lib/python3.13/site-packages/torch/__init__.py
# Becomes:
# /home/user/.venv/lib/python3.13/site-packages/torch/include
```

## Finding Python headers
```bash
python3-config --includes
# This outputs like: -I/usr/include/python3.10 -I/usr/include/python3.10
```

## Finding CUDA include path
```bash
# Check if CUDA is installed and find its location
which nvcc
nvcc --version

# Common CUDA locations:
# /usr/local/cuda/include
# /usr/cuda/include
# /opt/cuda/include

# You can also find it by:
find /usr -name "cuda.h" 2>/dev/null
```

## Finding functorch include path
```bash
# First check if functorch is installed
python -c "import functorch; print(functorch.__file__)"

# The include path is usually:
# Replace "__init__.py" with "include" in the path above
```

## Putting it all together
```bash
# Script to automatically generate clangd include paths
echo "Finding include paths for clangd..."

# Virtual environment path
VENV_PATH=$(python -c "import sys; print(sys.prefix)")
echo "Virtual environment: $VENV_PATH"

# PyTorch paths
TORCH_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")
echo "PyTorch include: $TORCH_PATH/include"
echo "PyTorch API include: $TORCH_PATH/include/torch/csrc/api/include"

# Python headers
PYTHON_INCLUDE=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
echo "Python include: $PYTHON_INCLUDE"

# CUDA path (if exists)
if command -v nvcc &> /dev/null; then
    CUDA_PATH=$(dirname $(dirname $(which nvcc)))
    echo "CUDA include: $CUDA_PATH/include"
fi
```