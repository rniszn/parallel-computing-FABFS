#!/usr/bin/env bash
set -uo pipefail

# M9: Master Analysis Pipeline
# Executes: load -> stats -> tables -> figures -> report_numbers
# Logs the SHA256 of results/levels.csv and all stage outputs to results/analysis/analysis.log

mkdir -p results/analysis results/tables results/figures report/tables report/figures

LOG_FILE="results/analysis/analysis.log"
echo "=== FABFS M9 Analysis Pipeline Run ===" | tee "$LOG_FILE"
echo "Timestamp: $(date -u)" | tee -a "$LOG_FILE"

LEVELS_FILE="results/levels.csv"
if [ ! -f "$LEVELS_FILE" ]; then
    echo "Error: $LEVELS_FILE not found! Cannot run analysis pipeline." | tee -a "$LOG_FILE"
    exit 1
fi

LEVELS_SHA256=$(sha256sum "$LEVELS_FILE" | awk '{print $1}')
echo "Input Dataset: $LEVELS_FILE" | tee -a "$LOG_FILE"
echo "SHA256: $LEVELS_SHA256" | tee -a "$LOG_FILE"
echo "Row Count: $(wc -l < "$LEVELS_FILE")" | tee -a "$LOG_FILE"
echo "--------------------------------------------------------" | tee -a "$LOG_FILE"

echo "[Stage 1/5] Running analysis/load.py..." | tee -a "$LOG_FILE"
python3 analysis/load.py | tee -a "$LOG_FILE"

echo "--------------------------------------------------------" | tee -a "$LOG_FILE"
echo "[Stage 2/5] Running analysis/stats.py..." | tee -a "$LOG_FILE"
stats_exit=0
python3 analysis/stats.py | tee -a "$LOG_FILE" || stats_exit=$?

echo "--------------------------------------------------------" | tee -a "$LOG_FILE"
echo "[Stage 3/5] Running analysis/tables.py..." | tee -a "$LOG_FILE"
python3 analysis/tables.py | tee -a "$LOG_FILE"

echo "--------------------------------------------------------" | tee -a "$LOG_FILE"
echo "[Stage 4/5] Running analysis/figures.py..." | tee -a "$LOG_FILE"
python3 analysis/figures.py | tee -a "$LOG_FILE"

echo "--------------------------------------------------------" | tee -a "$LOG_FILE"
echo "[Stage 5/5] Running analysis/report_numbers.py..." | tee -a "$LOG_FILE"
python3 analysis/report_numbers.py | tee -a "$LOG_FILE"

echo "--------------------------------------------------------" | tee -a "$LOG_FILE"
if [ "$stats_exit" -eq 0 ]; then
    echo "=== FABFS M9 Pipeline Completed Successfully (Pass/Fail: PASS) ===" | tee -a "$LOG_FILE"
else
    echo "=== FABFS M9 Pipeline Completed (Pass/Fail: FAILED or Incomplete Matrix, exit code: $stats_exit) ===" | tee -a "$LOG_FILE"
fi
echo "Analysis Log written to: $LOG_FILE" | tee -a "$LOG_FILE"

exit "$stats_exit"
