#!/usr/bin/env python3
"""
M9: Statistics and pass/fail check module.
Computes medians and 95% bootstrap confidence intervals with a fixed seed.
Evaluates the adaptive pass/fail criterion (within 5% of the better of Pure-Static and Pure-Dynamic).
"""

import sys
import numpy as np
import pandas as pd
from load import load_and_validate, EVALUATION_GRAPHS


class EvaluationStatus(str):
    """String subclass representing overall matrix evaluation status (PASS, FAIL, INCOMPLETE)."""
    def __bool__(self):
        return self == "PASS"


def compute_overall_status(pf_df):
    """
    Compute overall evaluation status from pass/fail DataFrame:
    - If any evaluation graph is NOT EVALUATED (missing policy runs), overall = INCOMPLETE.
    - If every evaluation graph is evaluated and at least one FAILs, overall = FAIL.
    - If every evaluation graph is evaluated and all PASS, overall = PASS.
    """
    if pf_df.empty:
        return EvaluationStatus("INCOMPLETE")

    n_not_eval = int((pf_df["status"] == "NOT EVALUATED").sum())
    n_fail = int((pf_df["status"] == "FAIL").sum())
    n_pass = int((pf_df["status"] == "PASS").sum())

    if n_not_eval > 0:
        return EvaluationStatus("INCOMPLETE")
    elif n_fail > 0:
        return EvaluationStatus("FAIL")
    elif n_pass > 0:
        return EvaluationStatus("PASS")
    else:
        return EvaluationStatus("INCOMPLETE")


def bootstrap_ci_median(data, num_resamples=2000, ci=95.0, seed=12345):
    """
    Compute a bootstrap confidence interval for the median using a fixed random seed.
    Returns (median, ci_lower, ci_upper).
    A cell with n = 1 round has no confidence interval (returns med, np.nan, np.nan).
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), np.nan, np.nan

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


def run_pass_fail_check(stats_df, threshold_ratio=1.05, eval_graphs=None):
    """
    Check the adaptive policy pass/fail criterion:
    On every evaluation graph, adaptive median total time is within 5% of
    the better of Pure-Static and Pure-Dynamic (median_adaptive <= 1.05 * min(median_static, median_dynamic)).

    Rules:
    - Only evaluation graphs are considered; any fixture graph (e.g. rmat10_seed1) is ignored.
    - For each graph, if any of the required policies (Adaptive, Pure-Static, Pure-Dynamic)
      has no complete runs, or the comparison needs a missing policy, the result is
      "NOT EVALUATED". It is never PASS or FAIL.
    - Status rules:
      - If any evaluation graph is NOT EVALUATED (missing policy runs) -> overall = INCOMPLETE.
      - If every evaluation graph is evaluated and at least one FAILs -> overall = FAIL.
      - If every evaluation graph is evaluated and all PASS -> overall = PASS.
    Returns (overall_status, results_df).
    """
    if eval_graphs is None:
        graphs_in_df = set(stats_df["graph"].unique()) if not stats_df.empty else set()
        # If any evaluation graph or known fixture is present, evaluate strictly the 8 evaluation graphs
        if graphs_in_df.intersection(EVALUATION_GRAPHS) or any("rmat10" in g for g in graphs_in_df):
            graphs_to_eval = EVALUATION_GRAPHS
        else:
            graphs_to_eval = sorted(graphs_in_df)
    else:
        graphs_to_eval = eval_graphs

    results = []
    for g in graphs_to_eval:
        if stats_df.empty or g not in stats_df["graph"].values:
            results.append({
                "graph": g,
                "adaptive_ms": np.nan,
                "static_ms": np.nan,
                "dynamic_ms": np.nan,
                "best_fixed_ms": np.nan,
                "ratio_to_best": np.nan,
                "diff_pct": np.nan,
                "passed": False,
                "status": "NOT EVALUATED",
                "note": "NOT EVALUATED"
            })
            continue

        gdf = stats_df[stats_df["graph"] == g].set_index("policy")

        has_adaptive = "Adaptive" in gdf.index and pd.notnull(gdf.loc["Adaptive", "median_ms"]) and gdf.loc["Adaptive", "n_runs"] > 0
        has_static = "Pure-Static" in gdf.index and pd.notnull(gdf.loc["Pure-Static", "median_ms"]) and gdf.loc["Pure-Static", "n_runs"] > 0
        has_dynamic = "Pure-Dynamic" in gdf.index and pd.notnull(gdf.loc["Pure-Dynamic", "median_ms"]) and gdf.loc["Pure-Dynamic", "n_runs"] > 0

        adapt_med = gdf.loc["Adaptive", "median_ms"] if "Adaptive" in gdf.index else np.nan
        static_med = gdf.loc["Pure-Static", "median_ms"] if "Pure-Static" in gdf.index else np.nan
        dyn_med = gdf.loc["Pure-Dynamic", "median_ms"] if "Pure-Dynamic" in gdf.index else np.nan

        # If any of the required policies is missing or has 0 complete runs
        if not (has_adaptive and has_static and has_dynamic):
            results.append({
                "graph": g,
                "adaptive_ms": adapt_med,
                "static_ms": static_med,
                "dynamic_ms": dyn_med,
                "best_fixed_ms": np.nan,
                "ratio_to_best": np.nan,
                "diff_pct": np.nan,
                "passed": False,
                "status": "NOT EVALUATED",
                "note": "NOT EVALUATED"
            })
            continue

        if static_med <= 0 or dyn_med <= 0 or np.isnan(static_med) or np.isnan(dyn_med):
            results.append({
                "graph": g,
                "adaptive_ms": adapt_med,
                "static_ms": static_med,
                "dynamic_ms": dyn_med,
                "best_fixed_ms": np.nan,
                "ratio_to_best": np.nan,
                "diff_pct": np.nan,
                "passed": False,
                "status": "NOT EVALUATED",
                "note": "NOT EVALUATED"
            })
            continue

        best_fixed = min(static_med, dyn_med)
        ratio = adapt_med / best_fixed if best_fixed > 0 else np.nan
        passed = bool(ratio <= threshold_ratio) if not np.isnan(ratio) else False
        status = "PASS" if passed else "FAIL"
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
            "status": status,
            "note": status
        })

    pf_df = pd.DataFrame(results)
    overall_status = compute_overall_status(pf_df)

    return overall_status, pf_df


def main():
    df, meta = load_and_validate()
    run_totals = compute_run_total_times(df)
    stats_df = compute_policy_graph_stats(run_totals, seed=12345)

    print("\n" + "="*70)
    print("FABFS Policy Total Time Statistics (Median and 95% Bootstrap CI)")
    print("="*70)
    display_graphs = [g for g in EVALUATION_GRAPHS if g in stats_df["graph"].unique()]
    if not display_graphs:
        display_graphs = sorted(stats_df["graph"].unique())

    for g in display_graphs:
        print(f"\nGraph: {g}")
        sub = stats_df[stats_df["graph"] == g]
        for _, row in sub.iterrows():
            n_runs = int(row['n_runs'])
            med_str = f"{row['median_ms']:>9.2f} ms (n={n_runs})"
            if n_runs == 1 or np.isnan(row['ci_lower_ms']) or np.isnan(row['ci_upper_ms']):
                ci_str = "[n=1, no CI]"
            else:
                ci_str = f"[95% CI: {row['ci_lower_ms']:>8.2f} - {row['ci_upper_ms']:>8.2f}]"
            print(f"  {row['policy']:<15} : {med_str}  {ci_str}")

    print("\n" + "="*70)
    print("Pass/Fail Evaluation Check: Adaptive <= 1.05 * min(Static, Dynamic)")
    print("="*70)
    overall_status, pf_df = run_pass_fail_check(stats_df)
    for _, row in pf_df.iterrows():
        status = row["status"]
        adapt_str = f"{row['adaptive_ms']:>8.2f}" if pd.notnull(row['adaptive_ms']) else "     N/A"
        best_str = f"{row['best_fixed_ms']:>8.2f}" if pd.notnull(row['best_fixed_ms']) else "     N/A"
        diff_str = f"{row['diff_pct']:>+7.2f}%" if pd.notnull(row['diff_pct']) else "    N/A"
        print(f"  {row['graph']:<15} : Adaptive={adapt_str} ms | BestFixed={best_str} ms | Diff={diff_str} | [{status}]")

    n_pass = int((pf_df["status"] == "PASS").sum()) if not pf_df.empty else 0
    n_fail = int((pf_df["status"] == "FAIL").sum()) if not pf_df.empty else 0
    n_not_eval = int((pf_df["status"] == "NOT EVALUATED").sum()) if not pf_df.empty else 0

    print("\n" + "="*70)
    print(f"Summary: {n_pass} PASS, {n_fail} FAIL, {n_not_eval} NOT EVALUATED")
    print(f"OVERALL STATUS: {overall_status}")
    print("="*70)

    if overall_status != "PASS":
        print(f"[stats.py] Evaluation check not passed (status: {overall_status}, {n_fail} FAIL, {n_not_eval} NOT EVALUATED).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
