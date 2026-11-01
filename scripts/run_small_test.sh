#!/usr/bin/env bash
set -euo pipefail

# M10: Evaluator Quick Re-run Script
# Matrix: 2 graphs x 4 policies x 2 rounds = 16 runs
# Validates execution correctness against M2 reference BFS
# Target runtime: Under 5 minutes

echo "=========================================================="
echo "FABFS Evaluator Quick Re-run (Small Matrix: 16 Runs)"
echo "Target Platform: NVIDIA RTX 4050 Laptop GPU + OpenMP CPU"
echo "=========================================================="

START_TIME=$(date +%s)
MANIFEST_FILE="results/small_test_manifest.json"
CSV_FILE="results/small_test_levels.csv"
RUNS_FILE="results/small_test_runs.jsonl"

mkdir -p results bin

# Clean previous test artifacts if present
rm -f "$MANIFEST_FILE" "$CSV_FILE" "$RUNS_FILE"

echo "[1/4] Generating small manifest (2 graphs x 4 policies x 2 rounds)..."
python3 - << 'EOF'
import json

graphs = ["roadNet-CA", "web-Google"]
policies = ["Adaptive", "Pure-Static", "Pure-Dynamic", "Oracle-Replay"]
rounds = [0, 1]

manifest = []
idx = 0
for r in rounds:
    for g in graphs:
        for p in policies:
            manifest.append({
                "run_id": f"small_run_{idx:03d}",
                "graph": g,
                "policy": p,
                "round": r
            })
            idx += 1

with open("results/small_test_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Generated {len(manifest)} runs in results/small_test_manifest.json")
EOF

echo "[2/4] Ensuring bin/bench_runner is up to date..."
make bin/bench_runner

echo "[3/4] Executing 16 evaluation runs with M2 correctness verification..."
./bin/bench_runner \
    --manifest="$MANIFEST_FILE" \
    --output="$CSV_FILE" \
    --runs="$RUNS_FILE" \
    --expected-count=16 \
    --allow-dirty

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "[4/4] Verifying outputs..."
COMPLETED_COUNT=$(wc -l < "$RUNS_FILE")
ROW_COUNT=$(wc -l < "$CSV_FILE")

echo "----------------------------------------------------------"
echo "Evaluator Re-run Results:"
echo "  Completed Runs: $COMPLETED_COUNT / 16"
echo "  CSV Levels Rows: $ROW_COUNT"
echo "  Elapsed Time:   ${ELAPSED} seconds (Target: < 300s)"
echo "----------------------------------------------------------"

if [ "$COMPLETED_COUNT" -ne 16 ]; then
    echo "ERROR: Expected 16 completed runs, got $COMPLETED_COUNT"
    exit 1
fi

if [ "$ELAPSED" -gt 300 ]; then
    echo "ERROR: Execution time exceeded 5-minute deadline: ${ELAPSED}s"
    exit 1
fi

echo "ALL 16 RUNS COMPLETED AND VERIFIED WITHIN 5 MINUTES. PASS."
echo "=========================================================="
exit 0
