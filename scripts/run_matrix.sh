#!/usr/bin/env bash
set -euo pipefail

# M8 Full Matrix Runner Entry Point
# Usage: ./scripts/run_matrix.sh [--dry-run] [--allow-dirty]

DRY_RUN=0
ALLOW_DIRTY_FLAG=""
ARGS=()

for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=1
        ARGS+=("--dry-run")
    elif [[ "$arg" == "--allow-dirty" ]]; then
        ALLOW_DIRTY_FLAG="--allow-dirty"
        ARGS+=("--allow-dirty")
    else
        ARGS+=("$arg")
    fi
done

# Step 1: Capture environment telemetry
./scripts/env_capture.sh $ALLOW_DIRTY_FLAG

# Step 2: Build runner binary
make bin/bench_runner

# Step 3: Execute benchmark runner
mkdir -p results
./bin/bench_runner "${ARGS[@]}"
