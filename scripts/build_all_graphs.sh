#!/usr/bin/env bash
set -euo pipefail

# Ensure results directory exists
mkdir -p results

# Remove any existing graphs.csv to regenerate cleanly
rm -f results/graphs.csv

echo "=== Building and Validating 8 Evaluation Graphs ==="

echo "1/12: roadNet-CA"
./build_graph roadNet-CA snap data/roadNet-CA.txt

echo "2/12: web-Google"
./build_graph web-Google snap data/web-Google.txt

echo "3/12: com-lj"
./build_graph com-lj snap data/com-lj.ungraph.txt

echo "4/12: com-orkut (with memory profiling)"
/usr/bin/time -v ./build_graph com-orkut snap data/com-orkut.ungraph.txt 2> results/com_orkut_time.txt
cat results/com_orkut_time.txt | grep -E "(Maximum resident set size|Elapsed \(wall clock\) time)"

echo "5/12: rmat_a0.25 (scale 20)"
./build_graph rmat_a0.25 rmat 0.25 20 16

echo "6/12: rmat_a0.45 (scale 20)"
./build_graph rmat_a0.45 rmat 0.45 20 16

echo "7/12: rmat_a0.55 (scale 20)"
./build_graph rmat_a0.55 rmat 0.55 20 16

echo "8/12: rmat_a0.65 (scale 20)"
./build_graph rmat_a0.65 rmat 0.65 20 16

echo "=== Building and Validating 4 Calibration Graphs ==="

echo "9/12: roadNet-PA"
./build_graph roadNet-PA snap data/roadNet-PA.txt

echo "10/12: com-amazon"
./build_graph com-amazon snap data/com-amazon.ungraph.txt

echo "11/12: com-dblp"
./build_graph com-dblp snap data/com-dblp.ungraph.txt

echo "12/12: rmat_s16_a0.45 (scale 16)"
./build_graph rmat_s16_a0.45 rmat 0.45 16 16

echo "=== ALL 12 GRAPHS BUILT AND VALIDATED SUCCESSFULLY ==="
