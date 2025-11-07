#!/bin/bash
#
# Launch multiple parallel training runs on a single GPU
#
# Usage:
#   bash launch_parallel_runs.sh [num_runs] [gpu_id]
#
# Example:
#   bash launch_parallel_runs.sh 8 0
#
# This will launch 8 training runs in parallel on GPU 0, each with a different
# random seed. Output goes to ../data/embeddings_128vocab_qweninit_run_NNN/

NUM_RUNS=${1:-8}
GPU_ID=${2:-0}
STEPS=${3:-10000}

echo "=================================================="
echo "Launching $NUM_RUNS parallel training runs on GPU $GPU_ID"
echo "=================================================="
echo ""

# Create log directory
mkdir -p ../logs

# Launch runs in background
for i in $(seq -f "%03g" 1 $NUM_RUNS); do
    SEED=$((42 + $(echo $i | sed 's/^0*//')))
    RUN_ID="run_$i"
    LOG_FILE="../logs/${RUN_ID}.log"

    echo "Starting $RUN_ID (seed=$SEED)..."

    uv run python train_parallel.py \
        --seed $SEED \
        --output $RUN_ID \
        --steps $STEPS \
        --gpu $GPU_ID \
        > $LOG_FILE 2>&1 &

    # Small delay to stagger startup
    sleep 2
done

echo ""
echo "✓ Launched $NUM_RUNS runs"
echo ""
echo "Monitor progress with:"
echo "  watch -n 5 'nvidia-smi && echo && tail -n 3 ../logs/run_*.log'"
echo ""
echo "Or check individual logs:"
echo "  tail -f ../logs/run_001.log"
echo ""
echo "Wait for all to complete:"
echo "  wait"
echo ""
