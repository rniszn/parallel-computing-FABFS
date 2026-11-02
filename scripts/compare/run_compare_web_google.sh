#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper for web-Google comparison
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"

python3 "$DIR/compare_one.py" \
    --graph="web-Google" \
    --graph-path="$REPO_ROOT/data/web-Google.txt" \
    "$@"
