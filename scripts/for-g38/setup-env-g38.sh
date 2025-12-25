#!/bin/bash
# g38 cluster environment setup for FIONA
# Usage: source setup-env-g38.sh

# Calculate FIONA_ROOT from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FIONA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Conda activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate fiona-undergraduate
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate fiona-undergraduate
fi

# RISCV toolchain
export RISCV="$HOME/riscv"
export PATH="$RISCV/bin:$PATH"

# FIONA Photonic
export FIONA_PHOTONIC_DIR="${FIONA_ROOT}/fiona-photonic"

# Python library path for libcustomext.so
PYTHON_LIB_PATH="$(python -c 'import sys; print(sys.prefix)')/lib"
export LD_LIBRARY_PATH="${PYTHON_LIB_PATH}:${LD_LIBRARY_PATH}"

# Spike path
export PATH="${FIONA_ROOT}/fiona-spikesim/build:$PATH"

echo "[INFO] g38 FIONA environment configured:"
echo "  - FIONA_ROOT: ${FIONA_ROOT}"
echo "  - RISCV: ${RISCV}"
echo "  - FIONA_PHOTONIC_DIR: ${FIONA_PHOTONIC_DIR}"
echo "  - Python: $(which python)"
echo "  - Spike: $(which spike)"
