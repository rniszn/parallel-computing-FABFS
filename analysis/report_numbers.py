#!/usr/bin/env python3
"""
M9: Report numbers generator.
Prints every single number quoted in the paper/report with explicit provenance
(source file and column). The report must quote only numbers from this script.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

from load import load_and_validate
from stats import compute_run_total_times, compute_policy_graph_stats, run_pass_fail_check


def print_provenance_number(name, value, unit, file_source, col_source, note=""):
    """Format and print an exact metric with clear source provenance."""
    note_str = f" ({note})" if note else ""
    print(f"  {name:<36} : {str(value):>14} {unit:<6} [Source: {file_source} | Col: {col_source}]{note_str}")


def main():
    print("=" * 100)
    print("FABFS EVALUATION REPORT: OFFICIAL REPRODUCIBLE NUMBERS TABLE")
    print("=" * 100)

    # 1. Environment & Calibration Configuration
    print("\n[SECTION 1: SYSTEM ENVIRONMENT & CALIBRATED PARAMETERS]")
    thresholds_path = "config/thresholds.json"
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresh = json.load(f)

    print_provenance_number("Calibration Commit Hash", thresh.get("calibration_commit"), "", thresholds_path, "calibration_commit")
    print_provenance_number("Threshold T_min (Frontier size)", thresh.get("T_min"), "vertices", thresholds_path, "T_min", "Below this, SEQ mode is selected")
    print_provenance_number("Threshold T_CV (Frontier skew)", thresh.get("T_CV"), "", thresholds_path, "T_CV", "Above this, DYNAMIC mode is selected")
    print_provenance_number("Threshold T_par (Parallel cutoff)", thresh.get("T_par"), "vertices", thresholds_path, "T_par", "Frontier size cutoff for static split")
    print_provenance_number("Static GPU Work Ratio (alpha)", thresh.get("alpha"), "", thresholds_path, "alpha", "GPU edge share in STATIC mode")
    print_provenance_number("CPU Edge Throughput (P_cpu)", thresh.get("P_cpu_edges_per_s"), "edges/s", thresholds_path, "P_cpu_edges_per_s")
    print_provenance_number("GPU Edge Throughput (P_gpu)", thresh.get("P_gpu_edges_per_s"), "edges/s", thresholds_path, "P_gpu_edges_per_s")
    print_provenance_number("GPU Kernel Launch Latency (L)", thresh.get("L_launch_us"), "us", thresholds_path, "L_launch_us")
    print_provenance_number("Steal Latency (C_steal)", thresh.get("C_steal_ns"), "ns", thresholds_path, "C_steal_ns")
    print_provenance_number("Dynamic Steal Chunk Max (C_max)", thresh.get("C_max"), "vertices", thresholds_path, "C_max")
    print_provenance_number("Dynamic Steal Chunk Min (C_min)", thresh.get("C_min"), "vertices", thresholds_path, "C_min")
    print_provenance_number("GPU Minimum Batch Size", thresh.get("GPU_MIN_BATCH"), "vertices", thresholds_path, "GPU_MIN_BATCH")

    # 2. Dataset Loading & Validation Telemetry
    print("\n[SECTION 2: MATRIX EXECUTION & DATASET VALIDATION]")
    df, meta = load_and_validate()
    print_provenance_number("Total Level Records", len(df), "rows", "results/levels.csv", "All rows")
    print_provenance_number("Completed Runs", meta["completed_runs_count"], "runs", "results/runs.jsonl", "status=complete")
    print_provenance_number("Timed-out Runs", meta["timeout_runs_count"], "runs", "results/runs.jsonl", "status=timeout")
    print_provenance_number("Evaluation Graphs Count", len(meta["graphs"]), "graphs", "results/levels.csv", "graph")
    print_provenance_number("Execution Policies Count", len(meta["policies"]), "policies", "results/levels.csv", "policy")

    git_commit_sample = df["git_commit"].iloc[0] if len(df) > 0 else "unknown"
    driver_version_sample = df["driver_version"].iloc[0] if len(df) > 0 else "unknown"
    print_provenance_number("Evaluation Git Commit", git_commit_sample, "", "results/levels.csv", "git_commit")
    print_provenance_number("NVIDIA CUDA Driver Version", driver_version_sample, "", "results/levels.csv", "driver_version")

    # 3. Policy Execution Times & Speedups
    print("\n[SECTION 3: POLICY PERFORMANCE & CONFIDENCE INTERVALS]")
    run_totals = compute_run_total_times(df)
    stats_df = compute_policy_graph_stats(run_totals, seed=12345)

    for g in sorted(stats_df["graph"].unique()):
        print(f"\n--- Graph: {g} ---")
        gdf = stats_df[stats_df["graph"] == g].set_index("policy")

        med_adapt = gdf.loc["Adaptive", "median_ms"] if "Adaptive" in gdf.index else np.nan
        lo_adapt = gdf.loc["Adaptive", "ci_lower_ms"] if "Adaptive" in gdf.index else np.nan
        hi_adapt = gdf.loc["Adaptive", "ci_upper_ms"] if "Adaptive" in gdf.index else np.nan

        med_static = gdf.loc["Pure-Static", "median_ms"] if "Pure-Static" in gdf.index else np.nan
        lo_static = gdf.loc["Pure-Static", "ci_lower_ms"] if "Pure-Static" in gdf.index else np.nan
        hi_static = gdf.loc["Pure-Static", "ci_upper_ms"] if "Pure-Static" in gdf.index else np.nan

        med_dyn = gdf.loc["Pure-Dynamic", "median_ms"] if "Pure-Dynamic" in gdf.index else np.nan
        lo_dyn = gdf.loc["Pure-Dynamic", "ci_lower_ms"] if "Pure-Dynamic" in gdf.index else np.nan
        hi_dyn = gdf.loc["Pure-Dynamic", "ci_upper_ms"] if "Pure-Dynamic" in gdf.index else np.nan

        med_oracle = gdf.loc["Oracle-Replay", "median_ms"] if "Oracle-Replay" in gdf.index else np.nan
        lo_oracle = gdf.loc["Oracle-Replay", "ci_lower_ms"] if "Oracle-Replay" in gdf.index else np.nan
        hi_oracle = gdf.loc["Oracle-Replay", "ci_upper_ms"] if "Oracle-Replay" in gdf.index else np.nan

        print_provenance_number(f"{g} Adaptive Median", f"{med_adapt:.2f}", "ms", "results/levels.csv", "time_us", f"95% CI: [{lo_adapt:.2f}, {hi_adapt:.2f}]")
        print_provenance_number(f"{g} Pure-Static Median", f"{med_static:.2f}", "ms", "results/levels.csv", "time_us", f"95% CI: [{lo_static:.2f}, {hi_static:.2f}]")
        print_provenance_number(f"{g} Pure-Dynamic Median", f"{med_dyn:.2f}", "ms", "results/levels.csv", "time_us", f"95% CI: [{lo_dyn:.2f}, {hi_dyn:.2f}]")
        print_provenance_number(f"{g} Oracle-Replay Median", f"{med_oracle:.2f}", "ms", "results/levels.csv", "time_us", f"95% CI: [{lo_oracle:.2f}, {hi_oracle:.2f}]")

        if not np.isnan(med_adapt) and not np.isnan(med_static) and med_adapt > 0:
            sp_static = med_static / med_adapt
            print_provenance_number(f"{g} Speedup over Pure-Static", f"{sp_static:.2f}x", "", "Derived", "Static/Adaptive")

        if not np.isnan(med_adapt) and not np.isnan(med_dyn) and med_adapt > 0:
            sp_dyn = med_dyn / med_adapt
            print_provenance_number(f"{g} Speedup over Pure-Dynamic", f"{sp_dyn:.2f}x", "", "Derived", "Dynamic/Adaptive")

        if not np.isnan(med_adapt) and not np.isnan(med_oracle) and med_oracle > 0:
            regret = (med_adapt - med_oracle) / med_oracle * 100.0
            print_provenance_number(f"{g} Oracle Regret", f"{regret:+.2f}%", "", "Derived", "(Adaptive-Oracle)/Oracle")

    # 4. Pass/Fail Decision Criteria
    print("\n[SECTION 4: ADAPTIVE PASS/FAIL CRITERIA (WITHIN 5% OF BEST FIXED)]")
    overall_pass, pf_df = run_pass_fail_check(stats_df)
    for _, row in pf_df.iterrows():
        g = row["graph"]
        diff_str = f"{row['diff_pct']:+.2f}%" if pd.notnull(row['diff_pct']) else "N/A"
        res_str = "PASS" if row["passed"] else "FAIL"
        print_provenance_number(f"{g} Pass/Fail Status", res_str, "", "results/tables/table3_pass_fail.csv", "passed", f"Diff from best fixed: {diff_str}")

    print_provenance_number("Overall Evaluation Matrix Result", "PASS" if overall_pass else "FAIL", "", "stats.py", "run_pass_fail_check")

    # 5. Telemetry & Overheads
    print("\n[SECTION 5: SYSTEM TELEMETRY & OVERHEAD METRICS]")
    gpu_idles = df["gpu_idle_us"].values
    if len(gpu_idles) > 0:
        print_provenance_number("Median GPU Proxy Idle Time", f"{np.median(gpu_idles):.1f}", "us", "results/levels.csv", "gpu_idle_us")
        print_provenance_number("Mean GPU Proxy Idle Time", f"{np.mean(gpu_idles):.1f}", "us", "results/levels.csv", "gpu_idle_us")
        print_provenance_number("95th Percentile GPU Idle Time", f"{np.percentile(gpu_idles, 95):.1f}", "us", "results/levels.csv", "gpu_idle_us")

    total_steals_att = df["steals_attempted"].sum()
    total_steals_succ = df["steals_succeeded"].sum()
    steal_succ_rate = (total_steals_succ / total_steals_att * 100.0) if total_steals_att > 0 else 0.0
    print_provenance_number("Total Steals Attempted", total_steals_att, "steals", "results/levels.csv", "steals_attempted")
    print_provenance_number("Total Steals Succeeded", total_steals_succ, "steals", "results/levels.csv", "steals_succeeded")
    print_provenance_number("Overall Steal Success Rate", f"{steal_succ_rate:.2f}%", "", "Derived", "steals_succeeded/steals_attempted")

    total_gpu_batches = df["gpu_batches"].sum()
    print_provenance_number("Total GPU Batches Dispatched", total_gpu_batches, "batches", "results/levels.csv", "gpu_batches")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
