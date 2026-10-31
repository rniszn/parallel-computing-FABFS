#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/env build
OUT="results/env/$(date +%F_%H%M).txt"

nvcc -O2 -o build/check_managed scripts/check_managed.cu

{
echo "== date"; date -Is
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.graphics --format=csv
echo "== nvidia-smi"; nvidia-smi
echo "== nvcc"; nvcc --version
echo "== gcc"; g++ --version | head -1
echo "== CPU"; lscpu | head -30
echo "== disk"; df -h .
echo "== git"; git rev-parse HEAD 2>/dev/null || echo "no commit yet"
echo "== managed memory"; ./build/check_managed
echo "== PASS"
} | tee "$OUT"
