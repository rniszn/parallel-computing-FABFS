#!/usr/bin/env python3
"""
M9: Master Provenance Numbers Table Generator.
Reads raw results from results/levels.csv, results/runs.jsonl, and config/thresholds.json,
and prints the authoritative numbers quoted across report tables and figures.
Every quoted number explicitly cites its source column and provenance calculation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from load import load_and_validate, EVALUATION_GRAPHS
from stats import compute_run_total_times, compute_policy_graph_stats, run_pass_fail_check


def print_provenance_number(label, val_str, unit="", source_file="", source_col="", note=""):
    unit_str = f" {unit}" if unit else ""
    src_info = f"[Source: {source_file} | Col: {source_col}]"
    if note:
        src_info += f" ({note})"
    print(f"  {label:<37}: {val_str:>15}{unit_str:<5} {src_info}")


def main():
    print("=" * 100)
    print("FABFS EVALUATION REPORT: OFFICIAL REPRODUCIBLE NUMBERS TABLE")
    print("=" * 100)

    # 1. Calibrated System Environment
    print("\n[SECTION 1: SYSTEM ENVIRONMENT & CALIBRATED PARAMETERS]")
    with open("config/thresholds.json", "r", encoding="utf-8") as f:
        thresh = json.load(f)

    print_provenance_number("Calibration Commit Hash", thresh.get("calibration_commit", "N/A"), "", "config/thresholds.json", "calibration_commit")
    print_provenance_number("Threshold T_min (Frontier size)", thresh.get("T_min", "N/A"), "vertices", "config/thresholds.json", "T_min", "Below this, SEQ mode is selected")
    print_provenance_number("Threshold T_CV (Frontier skew)", thresh.get("T_CV", "N/A"), "", "config/thresholds.json", "T_CV", "Above this, DYNAMIC mode is selected")
    print_provenance_number("Threshold T_par (Parallel cutoff)", thresh.get("T_par", "N/A"), "vertices", "config/thresholds.json", "T_par", "Frontier size cutoff for static split")
    print_provenance_number("Static GPU Work Ratio (alpha)", thresh.get("alpha", "N/A"), "", "config/thresholds.json", "alpha", "GPU edge share in STATIC mode")
    print_provenance_number("CPU Edge Throughput (P_cpu)", thresh.get("P_cpu_edges_per_s", "N/A"), "edges/s", "config/thresholds.json", "P_cpu_edges_per_s")
    print_provenance_number("GPU Edge Throughput (P_gpu)", thresh.get("P_gpu_edges_per_s", "N/A"), "edges/s", "config/thresholds.json", "P_gpu_edges_per_s")
    print_provenance_number("GPU Kernel Launch Latency (L)", thresh.get("L_launch_us", "N/A"), "us", "config/thresholds.json", "L_launch_us")
    print_provenance_number("Steal Latency (C_steal)", thresh.get("C_steal_ns", "N/A"), "ns", "config/thresholds.json", "C_steal_ns")
    print_provenance_number("Dynamic Steal Chunk Max (C_max)", thresh.get("C_max", "N/A"), "vertices", "config/thresholds.json", "C_max")
    print_provenance_number("Dynamic Steal Chunk Min (C_min)", thresh.get("C_min", "N/A"), "vertices", "config/thresholds.json", "C_min")
    print_provenance_number("GPU Minimum Batch Size", thresh.get("GPU_MIN_BATCH", "N/A"), "vertices", "config/thresholds.json", "GPU_MIN_BATCH")

    # 2. Evaluation Matrix Scale
    print("\n[SECTION 2: MATRIX EXECUTION & DATASET VALIDATION]")
    df, meta = load_and_validate()
    print_provenance_number("Total Level Records", len(df), "rows", "results/levels.csv", "All rows")
    print_provenance_number("Completed Runs", meta.get("completed_runs_count", "N/A"), "runs", "results/runs.jsonl", "status=complete")
    print_provenance_number("Timed-out Runs", meta.get("timeout_runs_count", "N/A"), "runs", "results/runs.jsonl", "status=timeout")
    print_provenance_number("Evaluation Graphs Count", df["graph"].nunique(), "graphs", "results/levels.csv", "graph")
    print_provenance_number("Execution Policies Count", df["policy"].nunique(), "policies", "results/levels.csv", "policy")

    if "git_commit" in df.columns:
        print_provenance_number("Evaluation Git Commit", df["git_commit"].iloc[0], "", "results/levels.csv", "git_commit")
    if "driver_version" in df.columns:
        print_provenance_number("NVIDIA CUDA Driver Version", df["driver_version"].iloc[0], "", "results/levels.csv", "driver_version")

    # 3. Policy Execution Times and Speedups
    print("\n[SECTION 3: POLICY PERFORMANCE & CONFIDENCE INTERVALS]")
    run_totals = compute_run_total_times(df)
    stats_df = compute_policy_graph_stats(run_totals, seed=12345)

    eval_graphs_in_stats = [g for g in EVALUATION_GRAPHS if g in stats_df["graph"].unique()]
    if not eval_graphs_in_stats:
        eval_graphs_in_stats = sorted(stats_df["graph"].unique())

    for g in eval_graphs_in_stats:
        print(f"\n--- Graph: {g} ---")
        gdf = stats_df[stats_df["graph"] == g].set_index("policy")

        med_adapt = gdf.loc["Adaptive", "median_ms"] if "Adaptive" in gdf.index else np.nan
        lo_adapt = gdf.loc["Adaptive", "ci_lower_ms"] if "Adaptive" in gdf.index else np.nan
        hi_adapt = gdf.loc["Adaptive", "ci_upper_ms"] if "Adaptive" in gdf.index else np.nan
        n_adapt = int(gdf.loc["Adaptive", "n_runs"]) if "Adaptive" in gdf.index else 0

        med_static = gdf.loc["Pure-Static", "median_ms"] if "Pure-Static" in gdf.index else np.nan
        lo_static = gdf.loc["Pure-Static", "ci_lower_ms"] if "Pure-Static" in gdf.index else np.nan
        hi_static = gdf.loc["Pure-Static", "ci_upper_ms"] if "Pure-Static" in gdf.index else np.nan
        n_static = int(gdf.loc["Pure-Static", "n_runs"]) if "Pure-Static" in gdf.index else 0

        med_dyn = gdf.loc["Pure-Dynamic", "median_ms"] if "Pure-Dynamic" in gdf.index else np.nan
        lo_dyn = gdf.loc["Pure-Dynamic", "ci_lower_ms"] if "Pure-Dynamic" in gdf.index else np.nan
        hi_dyn = gdf.loc["Pure-Dynamic", "ci_upper_ms"] if "Pure-Dynamic" in gdf.index else np.nan
        n_dyn = int(gdf.loc["Pure-Dynamic", "n_runs"]) if "Pure-Dynamic" in gdf.index else 0

        med_oracle = gdf.loc["Oracle-Replay", "median_ms"] if "Oracle-Replay" in gdf.index else np.nan
        lo_oracle = gdf.loc["Oracle-Replay", "ci_lower_ms"] if "Oracle-Replay" in gdf.index else np.nan
        hi_oracle = gdf.loc["Oracle-Replay", "ci_upper_ms"] if "Oracle-Replay" in gdf.index else np.nan
        n_oracle = int(gdf.loc["Oracle-Replay", "n_runs"]) if "Oracle-Replay" in gdf.index else 0

        ci_adapt_str = "n=1, no CI" if n_adapt == 1 or np.isnan(lo_adapt) else f"95% CI: [{lo_adapt:.2f}, {hi_adapt:.2f}]"
        ci_static_str = "n=1, no CI" if n_static == 1 or np.isnan(lo_static) else f"95% CI: [{lo_static:.2f}, {hi_static:.2f}]"
        ci_dyn_str = "n=1, no CI" if n_dyn == 1 or np.isnan(lo_dyn) else f"95% CI: [{lo_dyn:.2f}, {hi_dyn:.2f}]"
        ci_oracle_str = "n=1, no CI" if n_oracle == 1 or np.isnan(lo_oracle) else f"95% CI: [{lo_oracle:.2f}, {hi_oracle:.2f}]"

        print_provenance_number(f"{g} Adaptive Median", f"{med_adapt:.2f} (n={n_adapt})", "ms", "results/levels.csv", "time_us", ci_adapt_str)
        print_provenance_number(f"{g} Pure-Static Median", f"{med_static:.2f} (n={n_static})", "ms", "results/levels.csv", "time_us", ci_static_str)
        print_provenance_number(f"{g} Pure-Dynamic Median", f"{med_dyn:.2f} (n={n_dyn})", "ms", "results/levels.csv", "time_us", ci_dyn_str)
        print_provenance_number(f"{g} Oracle-Replay Median", f"{med_oracle:.2f} (n={n_oracle})", "ms", "results/levels.csv", "time_us", ci_oracle_str)

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
    overall_status, pf_df = run_pass_fail_check(stats_df)
    for _, row in pf_df.iterrows():
        g = row["graph"]
        diff_str = f"{row['diff_pct']:+.2f}%" if pd.notnull(row['diff_pct']) else "N/A"
        res_str = row["status"]
        print_provenance_number(f"{g} Pass/Fail Status", res_str, "", "results/tables/table3_pass_fail.csv", "status", f"Diff from best fixed: {diff_str}")

    print_provenance_number("Overall Evaluation Matrix Result", str(overall_status), "", "stats.py", "run_pass_fail_check")

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
