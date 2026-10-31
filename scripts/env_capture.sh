#!/usr/bin/env bash
set -euo pipefail

# M8 Environment Capture Script
# Usage: ./scripts/env_capture.sh [--allow-dirty] [output_file]

ALLOW_DIRTY=0
if [[ "${1:-}" == "--allow-dirty" ]]; then
    ALLOW_DIRTY=1
    shift
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H%M")
mkdir -p results/env
OUTPUT_FILE="${1:-results/env/${TIMESTAMP}.txt}"

# Check git status
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
DIRTY_STATUS=$(git status --porcelain 2>/dev/null || true)
IS_DIRTY=0

if [[ -n "$DIRTY_STATUS" ]]; then
    IS_DIRTY=1
    GIT_COMMIT="${GIT_COMMIT}-dirty"
fi

if [[ $IS_DIRTY -eq 1 && $ALLOW_DIRTY -eq 0 ]]; then
    echo "ERROR: Git working tree is dirty!" >&2
    echo "Refusing to start benchmark without clean commit." >&2
    echo "Pass --allow-dirty to override." >&2
    exit 1
fi

echo "Capturing environment telemetry to $OUTPUT_FILE..."

{
    echo "# FABFS Environment Telemetry Capture"
    echo "# Date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "# Hostname: $(hostname)"
    echo "git_commit: $GIT_COMMIT"
    echo "is_dirty: $IS_DIRTY"
    echo ""
    echo "--- System & CPU ---"
    uname -a
    if command -v lscpu >/dev/null 2>&1; then
        lscpu | grep -E "Model name|Architecture|CPU\(s\):|Thread\(s\) per core|Core\(s\) per socket" || true
    fi
    free -h || true
    echo ""
    echo "--- GPU & Driver Telemetry ---"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,driver_version,memory.total,power.management,power.draw,clocks.current.graphics,temperature.gpu --format=csv || true
    fi
    echo ""
    echo "--- CUDA Version ---"
    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version || true
    fi
    echo ""
    echo "--- Power Profile ---"
    if [[ -f /sys/class/power_supply/AC/online ]]; then
        echo "AC online: $(cat /sys/class/power_supply/AC/online)"
    fi
} > "$OUTPUT_FILE"

echo "Environment captured: $OUTPUT_FILE (Commit: $GIT_COMMIT)"
