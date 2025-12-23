#!/bin/bash
#
# run_parameter_sweep.sh - FIONA Photonic Parameter Sweep Experiments
#
# Usage:
#   ./run_parameter_sweep.sh [experiment] [mode]
#
# Experiments:
#   quant     - Quantization bit precision sweep
#   phase     - Phase error sweep
#   loss      - Insertion loss sweep
#   scenario  - Combined noise scenarios
#   all       - Run all experiments
#
# Mode:
#   quick     - 4 samples (fast, for verification)
#   full      - 872 samples (slow, for paper)
#
# Example:
#   ./run_parameter_sweep.sh quant quick
#   ./run_parameter_sweep.sh all full
#

set -e

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIONA_ROOT="/home/bspt628/fiona_undergraduate"
SPIKE_DIR="$FIONA_ROOT/fiona-spikesim"
PK_PATH="$FIONA_ROOT/chipyard/toolchains/riscv-tools/riscv-pk/build/pk"

# Parse arguments
EXPERIMENT=${1:-all}
MODE=${2:-quick}

if [ "$MODE" = "quick" ]; then
    ELF="$FIONA_ROOT/fiona-workload/build/sentiment_photonic.elf"
    echo "Mode: Quick (4 samples)"
else
    ELF="$FIONA_ROOT/fiona-workload/build/sentiment_benchmark_photonic.elf"
    echo "Mode: Full benchmark (872 samples)"
fi

# Output directory
RESULTS_DIR="$FIONA_ROOT/fiona-workload/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Source environment
source "$FIONA_ROOT/setup-env.sh"
cd "$SPIKE_DIR"

# Helper function
run_spike() {
    local name=$1
    shift
    echo "Running: $name"
    env "$@" spike --extension=fiona "$PK_PATH" "$ELF" 2>&1 | tee "$RESULTS_DIR/${name}.txt"
    echo ""
}

# Experiment 1: Quantization Precision
run_quant_sweep() {
    echo "=========================================="
    echo "Experiment 1: Quantization Precision Sweep"
    echo "=========================================="

    for bits in 2 3 4 5 6 7 8; do
        run_spike "quant_${bits}bit" \
            FIONA_PHOTONIC_MODEL=quantized \
            FIONA_QUANT_BITS=$bits
    done

    echo "Quantization sweep complete. Results in $RESULTS_DIR/quant_*.txt"
}

# Experiment 2: Phase Error
run_phase_sweep() {
    echo "=========================================="
    echo "Experiment 2: Phase Error Sweep"
    echo "=========================================="

    for sigma in 0.005 0.01 0.02 0.03 0.05 0.10; do
        run_spike "phase_${sigma}" \
            FIONA_PHOTONIC_MODEL=mzi_realistic \
            FIONA_PHASE_ERROR_SIGMA=$sigma
    done

    echo "Phase error sweep complete. Results in $RESULTS_DIR/phase_*.txt"
}

# Experiment 3: Insertion Loss
run_loss_sweep() {
    echo "=========================================="
    echo "Experiment 3: Insertion Loss Sweep"
    echo "=========================================="

    for loss in 0.1 0.2 0.3 0.5 0.7 1.0; do
        run_spike "loss_${loss}dB" \
            FIONA_PHOTONIC_MODEL=mzi_realistic \
            FIONA_INSERTION_LOSS_DB=$loss
    done

    echo "Insertion loss sweep complete. Results in $RESULTS_DIR/loss_*.txt"
}

# Experiment 4: Combined Scenarios
run_scenario_sweep() {
    echo "=========================================="
    echo "Experiment 4: Combined Noise Scenarios"
    echo "=========================================="

    # Ideal
    run_spike "scenario_ideal" \
        FIONA_PHOTONIC_MODEL=ideal

    # Optimistic
    run_spike "scenario_optimistic" \
        FIONA_PHOTONIC_MODEL=mzi_realistic \
        FIONA_PHASE_ERROR_SIGMA=0.005 \
        FIONA_INSERTION_LOSS_DB=0.2 \
        FIONA_CROSSTALK_DB=-35 \
        FIONA_QUANT_BITS=8

    # Realistic
    run_spike "scenario_realistic" \
        FIONA_PHOTONIC_MODEL=mzi_realistic \
        FIONA_PHASE_ERROR_SIGMA=0.02 \
        FIONA_INSERTION_LOSS_DB=0.3 \
        FIONA_CROSSTALK_DB=-25 \
        FIONA_QUANT_BITS=8

    # Pessimistic
    run_spike "scenario_pessimistic" \
        FIONA_PHOTONIC_MODEL=mzi_realistic \
        FIONA_PHASE_ERROR_SIGMA=0.05 \
        FIONA_INSERTION_LOSS_DB=0.5 \
        FIONA_CROSSTALK_DB=-20 \
        FIONA_QUANT_BITS=6

    # Worst-case
    run_spike "scenario_worstcase" \
        FIONA_PHOTONIC_MODEL=mzi_realistic \
        FIONA_PHASE_ERROR_SIGMA=0.10 \
        FIONA_INSERTION_LOSS_DB=1.0 \
        FIONA_CROSSTALK_DB=-15 \
        FIONA_QUANT_BITS=4

    echo "Scenario sweep complete. Results in $RESULTS_DIR/scenario_*.txt"
}

# Generate summary
generate_summary() {
    echo "=========================================="
    echo "Generating Summary"
    echo "=========================================="

    SUMMARY="$RESULTS_DIR/summary.txt"
    echo "FIONA Parameter Sweep Results" > "$SUMMARY"
    echo "Date: $(date)" >> "$SUMMARY"
    echo "Mode: $MODE" >> "$SUMMARY"
    echo "" >> "$SUMMARY"

    for f in "$RESULTS_DIR"/*.txt; do
        if [ "$f" != "$SUMMARY" ]; then
            echo "=== $(basename "$f" .txt) ===" >> "$SUMMARY"
            grep -E "(Results|Accuracy|correct)" "$f" >> "$SUMMARY" 2>/dev/null || echo "No results found" >> "$SUMMARY"
            echo "" >> "$SUMMARY"
        fi
    done

    echo "Summary saved to: $SUMMARY"
    cat "$SUMMARY"
}

# Main
case $EXPERIMENT in
    quant)
        run_quant_sweep
        ;;
    phase)
        run_phase_sweep
        ;;
    loss)
        run_loss_sweep
        ;;
    scenario)
        run_scenario_sweep
        ;;
    all)
        run_quant_sweep
        run_phase_sweep
        run_loss_sweep
        run_scenario_sweep
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Usage: $0 [quant|phase|loss|scenario|all] [quick|full]"
        exit 1
        ;;
esac

generate_summary

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
