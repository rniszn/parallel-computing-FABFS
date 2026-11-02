#!/usr/bin/env bash
set -euo pipefail

# Smoke test for M11 comparison harness
# Runs quick sanity pass on web-Google and verifies output tables and summary.json

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/.." && pwd)"

echo "=== Running FABFS Comparison Smoke Test (web-Google, --quick) ==="
cd "$REPO_ROOT"

# Ensure binary is built
make bin/compare_runner

# Run comparison script with --quick (2 warmup + 3 rounds)
./scripts/compare/run_compare_web_google.sh --quick

# Verify output files exist
SUMMARY_FILE="results/compare/web-Google/summary.json"
PER_LEVEL_FILE="results/compare/web-Google/per_level.csv"
LEVELS_FILE="results/compare/web-Google/levels.csv"

if [ ! -f "$SUMMARY_FILE" ]; then
    echo "ERROR: Missing $SUMMARY_FILE"
    exit 1
fi

if [ ! -f "$PER_LEVEL_FILE" ]; then
    echo "ERROR: Missing $PER_LEVEL_FILE"
    exit 1
fi

if [ ! -f "$LEVELS_FILE" ]; then
    echo "ERROR: Missing $LEVELS_FILE"
    exit 1
fi

echo "=== Smoke Test PASSED: web-Google comparison outputs verified ==="
exit 0
