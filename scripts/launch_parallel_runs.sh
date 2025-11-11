#!/bin/bash
#
# Launch multiple parallel training runs with a TOTAL step budget
#
# Usage:
#   bash launch_parallel_runs.sh <num_jobs> <total_steps> [gpu_id] [starting_run_number]
#
# Example:
#   bash launch_parallel_runs.sh 8 100000 0 101
#   Launches runs 101-108, each doing 12,500 steps (100k / 8)

NUM_JOBS=${1:-4}
TOTAL_STEPS=${2:-100000}
GPU_ID=${3:-0}
START_RUN=${4:-101}

# Calculate steps per job
STEPS_PER_JOB=$((TOTAL_STEPS / NUM_JOBS))

echo "=================================================="
echo "Launching $NUM_JOBS parallel training runs on GPU $GPU_ID"
echo "Total step budget: $TOTAL_STEPS"
echo "Steps per job: $STEPS_PER_JOB"
echo "Run IDs: $(printf "%03d" $START_RUN)-$(printf "%03d" $((START_RUN + NUM_JOBS - 1)))"
echo "=================================================="
echo ""

# Create log directory
mkdir -p ../logs

# Launch runs in background
for i in $(seq -f "%03g" $START_RUN $((START_RUN + NUM_JOBS - 1))); do
    SEED=$((42 + $(echo $i | sed 's/^0*//')))
    RUN_ID="run_$i"
    LOG_FILE="../logs/${RUN_ID}.log"

    echo "Starting $RUN_ID (seed=$SEED, steps=$STEPS_PER_JOB)..."

    uv run python train_parallel.py \
        --seed $SEED \
        --output $RUN_ID \
        --steps $STEPS_PER_JOB \
        --gpu $GPU_ID \
        > $LOG_FILE 2>&1 &

    # Small delay to stagger startup
    sleep 2
done

echo ""
echo "✓ Launched $NUM_JOBS runs"
echo ""
echo "Monitor progress with:"
echo "  watch -n 5 'nvidia-smi && echo && tail -n 3 ../logs/run_*.log'"
echo ""
echo "Or check individual logs:"
echo "  tail -f ../logs/run_$(printf "%03d" $START_RUN).log"
echo ""
echo "Wait for all to complete:"
echo "  wait"
echo ""
