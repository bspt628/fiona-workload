#!/bin/bash
#SBATCH -p defq
#SBATCH -n 1
#SBATCH -J fiona-spike
#SBATCH -o fiona-%J.out
#SBATCH -e fiona-%J.err

# Usage: sbatch run_spike.sh <app_name>
# Example: sbatch run_spike.sh benchmark_transformer

APP_NAME=${1:-sentiment_benchmark}

# Calculate FIONA_ROOT from submit directory (scripts/for-g38 -> fiona-workload -> fiona_undergraduate)
# SLURM_SUBMIT_DIR is the directory where sbatch was executed
FIONA_ROOT="$(cd "${SLURM_SUBMIT_DIR}/../../.." && pwd)"

PK_PATH=${FIONA_ROOT}/chipyard/toolchains/riscv-tools/riscv-pk/build/pk
ELF_PATH=${FIONA_ROOT}/fiona-workload/build/${APP_NAME}.elf

source ${FIONA_ROOT}/setup-env.sh
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
