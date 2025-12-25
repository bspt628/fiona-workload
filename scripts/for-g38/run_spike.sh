#!/bin/bash
#SBATCH -p defq
#SBATCH -n 1
#SBATCH -J fiona-spike
#SBATCH -o fiona-%J.out
#SBATCH -e fiona-%J.err

# Usage: sbatch run_spike.sh <app_name>
# Example: sbatch run_spike.sh benchmark_transformer

APP_NAME=${1:-sentiment_benchmark}

# Source g38-specific environment setup
source "${SLURM_SUBMIT_DIR}/setup-env-g38.sh"

# Paths (FIONA_ROOT is set by setup-env-g38.sh)
PK_PATH="${RISCV}/riscv64-unknown-elf/bin/pk"
ELF_PATH="${FIONA_ROOT}/fiona-workload/build/${APP_NAME}.elf"

cd ${FIONA_ROOT}/fiona-spikesim

echo "=== FIONA Spike Simulation ==="
echo "App: ${APP_NAME}"
echo "ELF: ${ELF_PATH}"
echo "Start: $(date)"
echo "=============================="

if [ ! -f "${ELF_PATH}" ]; then
    echo "Error: ELF file not found: ${ELF_PATH}"
    exit 1
fi

spike --extension=fiona ${PK_PATH} ${ELF_PATH}

echo "=============================="
echo "End: $(date)"
