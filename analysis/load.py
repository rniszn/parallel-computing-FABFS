#!/usr/bin/env python3
"""
M9: Data loading and validation module.
Verifies the 28-column contract, run integrity against runs.jsonl,
input dataset hashes, and commit consistency between gate1.txt and thresholds.json.
"""

import os
import sys
import json
import pandas as pd

CONTRACT_COLUMNS = [
    "run_id",
    "graph",
    "policy",
    "round",
    "level",
    "frontier_size",
    "edge_work",
    "mu",
    "cv",
    "dmax",
    "mode_chosen",
    "chunk_size",
    "time_us",
    "steals_attempted",
    "steals_succeeded",
    "gpu_batches",
    "correct",
    "git_commit",
    "driver_version",
    "threads",
    "alpha",
    "gpu_min_batch",
    "num_chunks",
    "gpu_batch_vertices",
    "t_cpu_us",
    "t_gpu_us",
    "t_merge_us",
    "gpu_idle_us"
]


def check_header(columns):
    """Check the CSV header against the 28-column contract."""
    cols = list(columns)
    if cols != CONTRACT_COLUMNS:
        missing = [c for c in CONTRACT_COLUMNS if c not in cols]
        extra = [c for c in cols if c not in CONTRACT_COLUMNS]
        msg = f"Header contract violation: expected {len(CONTRACT_COLUMNS)} columns, got {len(cols)}."
        if missing:
            msg += f" Missing: {missing}."
        if extra:
            msg += f" Extra: {extra}."
        if not missing and not extra and cols != CONTRACT_COLUMNS:
            msg += f" Column ordering mismatch: {cols} vs {CONTRACT_COLUMNS}."
        raise ValueError(msg)
    return True


def load_runs_jsonl(runs_jsonl_path):
    """Load runs from runs.jsonl and categorize into complete and timeout."""
    completed_runs = set()
    timeout_runs = set()
    all_runs = {}

    if not os.path.exists(runs_jsonl_path):
        raise FileNotFoundError(f"runs.jsonl not found at {runs_jsonl_path}")

    with open(runs_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            run_id = entry.get("run_id")
            status = entry.get("status")
            all_runs[run_id] = entry
            if status == "complete":
                completed_runs.add(run_id)
            elif status == "timeout":
                timeout_runs.add(run_id)

    return completed_runs, timeout_runs, all_runs


def check_input_hashes(graphs, inputs_path):
    """Check that every graph in levels.csv appears in the inputs hash file."""
    if not os.path.exists(inputs_path):
        raise FileNotFoundError(f"Input hashes file not found at {inputs_path}")

    known_graphs = set()
    with open(inputs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts:
                known_graphs.add(parts[0])

    for g in graphs:
        if g not in known_graphs:
            raise ValueError(f"Graph '{g}' not found in inputs hash file {inputs_path}")
    return True


def check_gate1_thresholds_commit(gate1_path, thresholds_path):
    """Check that the gate1 record and the frozen thresholds commit hash match."""
    if not os.path.exists(gate1_path):
        raise FileNotFoundError(f"gate1.txt not found at {gate1_path}")
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"thresholds.json not found at {thresholds_path}")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresh = json.load(f)
    thresh_commit = thresh.get("calibration_commit", "").strip()

    gate1_commit = None
    with open(gate1_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# Git Commit:"):
                gate1_commit = line.split(":", 1)[1].strip()
                break

    if not thresh_commit:
        raise ValueError(f"No calibration_commit found in {thresholds_path}")
    if not gate1_commit:
        raise ValueError(f"No '# Git Commit:' line found in {gate1_path}")

    if thresh_commit != gate1_commit:
        raise ValueError(
            f"Commit mismatch: gate1 commit ({gate1_commit}) != thresholds commit ({thresh_commit})"
        )

    return thresh_commit


def load_and_validate(
    levels_csv_path="results/levels.csv",
    runs_jsonl_path="results/runs.jsonl",
    inputs_path="results/env/inputs.txt",
    gate1_path="results/gate1.txt",
    thresholds_path="config/thresholds.json",
):
    """Load levels.csv and enforce all integrity and contract checks."""
    if not os.path.exists(levels_csv_path):
        raise FileNotFoundError(f"levels.csv not found at {levels_csv_path}")

    df = pd.read_csv(levels_csv_path)

    # 1. Header contract check
    check_header(df.columns)

    # 2. Runs status check
    completed_runs, timeout_runs, all_runs = load_runs_jsonl(runs_jsonl_path)

    # Check for timeout runs and exclude them
    if timeout_runs:
        print(f"[load.py] Warning: Found {len(timeout_runs)} timed-out run(s): {sorted(list(timeout_runs))}")
        print(f"[load.py] Excluding timed-out runs from statistics.")
        df = df[~df["run_id"].isin(timeout_runs)].copy()

    # Verify that every remaining run_id is complete
    unique_runs = set(df["run_id"].unique())
    incomplete_runs = unique_runs - completed_runs
    if incomplete_runs:
        raise ValueError(
            f"Integrity check failed: {len(incomplete_runs)} run(s) in levels.csv are not marked 'complete' in runs.jsonl: {incomplete_runs}"
        )

    # 3. Input hashes check
    graphs_in_data = set(df["graph"].unique())
    check_input_hashes(graphs_in_data, inputs_path)

    # 4. Gate 1 & Thresholds commit check
    commit_hash = check_gate1_thresholds_commit(gate1_path, thresholds_path)

    # 5. Check no warmup runs appear (rounds should be non-negative evaluation rounds)
    if (df["round"] < 0).any():
        raise ValueError("Warmup runs detected in levels.csv (found negative round indices)")

    meta = {
        "commit": commit_hash,
        "completed_runs_count": len(unique_runs),
        "timeout_runs_count": len(timeout_runs),
        "total_rows": len(df),
        "graphs": sorted(list(graphs_in_data)),
        "policies": sorted(list(df["policy"].unique())),
    }

    return df, meta


if __name__ == "__main__":
    try:
        df, meta = load_and_validate()
        print(f"Data successfully validated and loaded:")
        print(f"  Rows: {meta['total_rows']}")
        print(f"  Completed Runs: {meta['completed_runs_count']}")
        print(f"  Timeout Runs: {meta['timeout_runs_count']}")
        print(f"  Graphs ({len(meta['graphs'])}): {meta['graphs']}")
        print(f"  Policies ({len(meta['policies'])}): {meta['policies']}")
        print(f"  Gate 1 Commit: {meta['commit']}")
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
