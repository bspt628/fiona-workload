#!/bin/bash
# Parallel execution script for sentiment_benchmark_photonic
# Usage: ./run_photonic_parallel.sh [num_parallel] [total_samples]

NUM_PARALLEL=${1:-8}      # Default: 8 parallel processes
TOTAL_SAMPLES=${2:-872}   # Default: all 872 samples

# Calculate samples per worker
SAMPLES_PER_WORKER=$((TOTAL_SAMPLES / NUM_PARALLEL))
REMAINDER=$((TOTAL_SAMPLES % NUM_PARALLEL))

echo "============================================"
echo "  Parallel Photonic Benchmark"
echo "============================================"
echo "  Total samples:    $TOTAL_SAMPLES"
echo "  Parallel workers: $NUM_PARALLEL"
echo "  Samples/worker:   ~$SAMPLES_PER_WORKER"
echo "============================================"

# Create output directory
OUTDIR="parallel_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTDIR

# Set photonic model to ideal for fastest execution
export FIONA_PHOTONIC_MODEL=ideal

# Paths
SPIKE="spike"
PK="$RISCV/riscv64-unknown-elf/bin/pk"
ELF="../build/sentiment_benchmark_photonic.elf"

# Launch parallel workers
pids=()
start=0
for ((i=0; i<NUM_PARALLEL; i++)); do
    # Distribute remainder samples to first workers
    if [ $i -lt $REMAINDER ]; then
        count=$((SAMPLES_PER_WORKER + 1))
    else
        count=$SAMPLES_PER_WORKER
    fi

    end=$((start + count))

    echo "Starting worker $i: samples $start-$end"

    # Run in background, save output
    $SPIKE --extension=fiona $PK $ELF $start $end > "$OUTDIR/worker_${i}_${start}_${end}.log" 2>&1 &
    pids+=($!)

    start=$end
done

echo ""
echo "All workers started. Waiting for completion..."
echo "Logs: $OUTDIR/"
echo ""

# Wait for all workers
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All workers completed!"
echo ""

# Aggregate results
echo "============================================"
echo "  Aggregated Results"
echo "============================================"

total_correct=0
total_samples=0

for logfile in $OUTDIR/worker_*.log; do
    # Extract accuracy line
    acc_line=$(grep "Accuracy:" $logfile | tail -1)
    if [ -n "$acc_line" ]; then
        # Parse correct/total from format: Accuracy: XX.XX% (correct/total)
        correct=$(echo "$acc_line" | sed -n 's/.*(\([0-9]*\)\/\([0-9]*\)).*/\1/p')
        samples=$(echo "$acc_line" | sed -n 's/.*(\([0-9]*\)\/\([0-9]*\)).*/\2/p')

        if [ -n "$correct" ] && [ -n "$samples" ]; then
            total_correct=$((total_correct + correct))
            total_samples=$((total_samples + samples))
            echo "  $(basename $logfile): $correct/$samples correct"
        fi
    fi
done

if [ $total_samples -gt 0 ]; then
    accuracy=$(echo "scale=2; 100 * $total_correct / $total_samples" | bc)
    echo ""
    echo "Total: $total_correct/$total_samples = ${accuracy}%"
fi

echo ""
echo "Done!"
