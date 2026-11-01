#!/usr/bin/env python3
"""
M9: Statistics and pass/fail check module.
Computes medians and 95% bootstrap confidence intervals with a fixed seed.
Evaluates the adaptive pass/fail criterion (within 5% of the better of Pure-Static and Pure-Dynamic).
"""

import sys
import numpy as np
import pandas as pd
from load import load_and_validate


def bootstrap_ci_median(data, num_resamples=2000, ci=95.0, seed=12345):
    """
    Compute a bootstrap confidence interval for the median using a fixed random seed.
    Returns (median, ci_lower, ci_upper).
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    # Generate bootstrap samples
    resamples = rng.choice(arr, size=(num_resamples, len(arr)), replace=True)
    resample_medians = np.median(resamples, axis=1)

    alpha = 100.0 - ci
    lower = float(np.percentile(resample_medians, alpha / 2.0))
    upper = float(np.percentile(resample_medians, 100.0 - alpha / 2.0))
    med = float(np.median(arr))

    return med, lower, upper


def compute_run_total_times(df):
    """
    Compute total time per run by summing time_us across all levels of that run.
    Returns a DataFrame with columns: [run_id, graph, policy, round, total_time_us, total_time_ms].
    """
    run_totals = df.groupby(["run_id", "graph", "policy", "round"], as_index=False)["time_us"].sum()
    run_totals.rename(columns={"time_us": "total_time_us"}, inplace=True)
    run_totals["total_time_ms"] = run_totals["total_time_us"] / 1000.0
    return run_totals


def compute_policy_graph_stats(run_totals, seed=12345):
    """
    Compute median total time and 95% bootstrap CI for each (graph, policy).
    Returns a DataFrame.
    """
    records = []
    graphs = sorted(run_totals["graph"].unique())
    policies = sorted(run_totals["policy"].unique())

    for g in graphs:
        for p in policies:
            sub = run_totals[(run_totals["graph"] == g) & (run_totals["policy"] == p)]
            times = sub["total_time_ms"].values
            n_runs = len(times)
            med, lo, hi = bootstrap_ci_median(times, num_resamples=2000, ci=95.0, seed=seed)
            records.append({
                "graph": g,
                "policy": p,
                "n_runs": n_runs,
                "median_ms": med,
                "ci_lower_ms": lo,
                "ci_upper_ms": hi
            })

    return pd.DataFrame(records)


def run_pass_fail_check(stats_df, threshold_ratio=1.05):
    """
    Check the adaptive policy pass/fail criterion:
    On every evaluation graph, adaptive median total time is within 5% of
    the better of Pure-Static and Pure-Dynamic (median_adaptive <= 1.05 * min(median_static, median_dynamic)).
    Returns (overall_passed, results_list).
    """
    results = []
    overall_pass = True

    graphs = sorted(stats_df["graph"].unique())
    for g in graphs:
        gdf = stats_df[stats_df["graph"] == g].set_index("policy")
        
        has_adaptive = "Adaptive" in gdf.index
        has_static = "Pure-Static" in gdf.index
        has_dynamic = "Pure-Dynamic" in gdf.index

        if not (has_adaptive and has_static and has_dynamic):
            results.append({
                "graph": g,
                "adaptive_ms": gdf.loc["Adaptive", "median_ms"] if has_adaptive else np.nan,
                "static_ms": gdf.loc["Pure-Static", "median_ms"] if has_static else np.nan,
                "dynamic_ms": gdf.loc["Pure-Dynamic", "median_ms"] if has_dynamic else np.nan,
                "best_fixed_ms": np.nan,
                "ratio_to_best": np.nan,
                "passed": False,
                "note": "Incomplete policy data"
            })
            overall_pass = False
            continue

        adapt_med = gdf.loc["Adaptive", "median_ms"]
        static_med = gdf.loc["Pure-Static", "median_ms"]
        dyn_med = gdf.loc["Pure-Dynamic", "median_ms"]

        best_fixed = min(static_med, dyn_med)
        ratio = adapt_med / best_fixed if best_fixed > 0 else np.nan
        passed = (ratio <= threshold_ratio) if not np.isnan(ratio) else False

        if not passed:
            overall_pass = False

        diff_pct = (ratio - 1.0) * 100.0 if not np.isnan(ratio) else np.nan

        results.append({
            "graph": g,
            "adaptive_ms": adapt_med,
            "static_ms": static_med,
            "dynamic_ms": dyn_med,
            "best_fixed_ms": best_fixed,
            "ratio_to_best": ratio,
            "diff_pct": diff_pct,
            "passed": passed,
            "note": "PASS" if passed else "FAIL"
        })

    return overall_pass, pd.DataFrame(results)


def main():
    df, meta = load_and_validate()
    run_totals = compute_run_total_times(df)
    stats_df = compute_policy_graph_stats(run_totals, seed=12345)

    print("\n" + "="*70)
    print("FABFS Policy Total Time Statistics (Median and 95% Bootstrap CI)")
    print("="*70)
    for g in sorted(stats_df["graph"].unique()):
        print(f"\nGraph: {g}")
        sub = stats_df[stats_df["graph"] == g]
        for _, row in sub.iterrows():
            print(f"  {row['policy']:<15} : {row['median_ms']:>9.3f} ms  [95% CI: {row['ci_lower_ms']:>8.3f} - {row['ci_upper_ms']:>8.3f}]  (N={row['n_runs']})")

    print("\n" + "="*70)
    print("Pass/Fail Evaluation Check: Adaptive <= 1.05 * min(Static, Dynamic)")
    print("="*70)
    overall_pass, pf_df = run_pass_fail_check(stats_df)
    for _, row in pf_df.iterrows():
        status = "PASS" if row["passed"] else "FAIL"
        diff_str = f"{row['diff_pct']:+.2f}%" if not np.isnan(row['diff_pct']) else "N/A"
        print(f"  {row['graph']:<15} : Adaptive={row['adaptive_ms']:>8.2f} ms | BestFixed={row['best_fixed_ms']:>8.2f} ms | Diff={diff_str:>7} | [{status}]")

    print("\n" + "="*70)
    print(f"OVERALL STATUS: {'PASS' if overall_pass else 'FAIL'}")
    print("="*70)

    if not overall_pass:
        print("[stats.py] Evaluation check failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
