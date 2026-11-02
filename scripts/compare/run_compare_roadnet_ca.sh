#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper for roadNet-CA comparison
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"

python3 "$DIR/compare_one.py" \
    --graph="roadNet-CA" \
    --graph-path="$REPO_ROOT/data/roadNet-CA.txt" \
    "$@"
