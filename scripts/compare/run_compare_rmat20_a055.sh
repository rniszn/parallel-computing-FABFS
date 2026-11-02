#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper for R-MAT scale 20, a=0.55 comparison
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"

python3 "$DIR/compare_one.py" \
    --graph="rmat_a0.55" \
    --graph-path="$REPO_ROOT/data/rmat_a0.55.txt" \
    "$@"
