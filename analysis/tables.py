#!/usr/bin/env python3
"""
M9: Tables generator.
Generates Table 1 (Graph Properties), Table 2 (Median Total Times with CI),
Table 3 (Pass/Fail Verification), and Table 4 (Completed Run Counts Matrix).
Outputs both CSV and LaTeX formats to results/tables/ and report/tables/.
"""

import os
import sys
import numpy as np
import pandas as pd
from load import load_and_validate
from stats import compute_run_total_times, compute_policy_graph_stats, run_pass_fail_check


def generate_table1_graph_properties(graphs_csv_path="results/graphs.csv", out_dir="results/tables"):
    """
    Table 1: Graph properties from M1 graphs.csv.
    |V|, |E| as undirected counts (m_directed / 2), mean degree, max degree, degree CV.
    """
    if not os.path.exists(graphs_csv_path):
        raise FileNotFoundError(f"{graphs_csv_path} not found")

    gdf = pd.read_csv(graphs_csv_path)

    # Filter for evaluation graphs if present, or format all
    eval_graphs = [
        "roadNet-CA", "web-Google", "com-lj", "com-orkut",
        "rmat_a0.25", "rmat_a0.45", "rmat_a0.55", "rmat_a0.65"
    ]
    sub = gdf[gdf["graph"].isin(eval_graphs)].copy()
    if sub.empty:
        sub = gdf.copy()

    # Undirected |E|
    sub["m_undirected"] = (sub["m_directed"] // 2).astype(int)

    table1 = pd.DataFrame({
        "Graph": sub["graph"],
        "|V|": sub["n"].apply(lambda x: f"{x:,}"),
        "|E|": sub["m_undirected"].apply(lambda x: f"{x:,}"),
        "Mean Degree": sub["mean_degree"].apply(lambda x: f"{x:.2f}"),
        "Max Degree": sub["max_degree"].apply(lambda x: f"{x:,}"),
        "Degree CV": sub["degree_cv"].apply(lambda x: f"{x:.2f}")
    })

    # CSV output
    csv_path = os.path.join(out_dir, "table1_graph_properties.csv")
    table1.to_csv(csv_path, index=False)

    # LaTeX output
    tex_path = os.path.join(out_dir, "table1_graph_properties.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Table 1: Benchmark Evaluation Graph Properties\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\hline\n")
        f.write("Graph & $|V|$ & $|E|$ & Mean Degree & Max Degree & Degree CV \\\\\n")
        f.write("\\hline\n")
        for _, r in table1.iterrows():
            f.write(f"{r['Graph']} & {r['|V|']} & {r['|E|']} & {r['Mean Degree']} & {r['Max Degree']} & {r['Degree CV']} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    return table1


def generate_table2_policy_times(stats_df, out_dir="results/tables"):
    """
    Table 2: Median total time with 95% bootstrap CI for each policy and graph.
    """
    graphs = sorted(stats_df["graph"].unique())
    policies = ["Adaptive", "Pure-Static", "Pure-Dynamic", "Oracle-Replay"]

    rows = []
    for g in graphs:
        row = {"Graph": g}
        for p in policies:
            match = stats_df[(stats_df["graph"] == g) & (stats_df["policy"] == p)]
            if not match.empty:
                med = match.iloc[0]["median_ms"]
                lo = match.iloc[0]["ci_lower_ms"]
                hi = match.iloc[0]["ci_upper_ms"]
                row[p] = f"{med:.2f} [{lo:.2f}, {hi:.2f}]"
            else:
                row[p] = "N/A"
        rows.append(row)

    table2 = pd.DataFrame(rows)

    # CSV output
    csv_path = os.path.join(out_dir, "table2_policy_times.csv")
    table2.to_csv(csv_path, index=False)

    # LaTeX output
    tex_path = os.path.join(out_dir, "table2_policy_times.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Table 2: Median Total Traversal Time (ms) with 95\\% Bootstrap CI\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Graph & Adaptive [95\\% CI] & Pure-Static [95\\% CI] & Pure-Dynamic [95\\% CI] & Oracle-Replay [95\\% CI] \\\\\n")
        f.write("\\hline\n")
        for _, r in table2.iterrows():
            f.write(f"{r['Graph']} & {r['Adaptive']} & {r['Pure-Static']} & {r['Pure-Dynamic']} & {r['Oracle-Replay']} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    return table2


def generate_table3_pass_fail(pf_df, out_dir="results/tables"):
    """
    Table 3: Pass/fail results per graph.
    """
    table3 = pd.DataFrame({
        "Graph": pf_df["graph"],
        "Adaptive (ms)": pf_df["adaptive_ms"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"),
        "Static (ms)": pf_df["static_ms"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"),
        "Dynamic (ms)": pf_df["dynamic_ms"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"),
        "Best Fixed (ms)": pf_df["best_fixed_ms"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"),
        "Diff (%)": pf_df["diff_pct"].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A"),
        "Result": pf_df["passed"].apply(lambda x: "PASS" if x else "FAIL")
    })

    # CSV output
    csv_path = os.path.join(out_dir, "table3_pass_fail.csv")
    table3.to_csv(csv_path, index=False)

    # LaTeX output
    tex_path = os.path.join(out_dir, "table3_pass_fail.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Table 3: Pass/Fail Evaluation Criteria: Adaptive <= 1.05 * min(Static, Dynamic)\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("Graph & Adaptive (ms) & Pure-Static (ms) & Pure-Dynamic (ms) & Best Fixed (ms) & Diff (\\%) & Status \\\\\n")
        f.write("\\hline\n")
        for _, r in table3.iterrows():
            f.write(f"{r['Graph']} & {r['Adaptive (ms)']} & {r['Static (ms)']} & {r['Dynamic (ms)']} & {r['Best Fixed (ms)']} & {r['Diff (%)']} & {r['Result']} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    return table3


def generate_table4_run_counts(run_totals, out_dir="results/tables"):
    """
    Table 4: Completed run counts per cell (Policy x Graph) to verify matrix completeness.
    """
    counts = run_totals.groupby(["graph", "policy"]).size().unstack(fill_value=0)
    counts.reset_index(inplace=True)
    counts.rename(columns={"graph": "Graph"}, inplace=True)

    # Reorder columns
    desired_cols = ["Graph", "Adaptive", "Pure-Static", "Pure-Dynamic", "Oracle-Replay"]
    actual_cols = ["Graph"] + [c for c in desired_cols if c in counts.columns and c != "Graph"]
    table4 = counts[actual_cols]

    # CSV output
    csv_path = os.path.join(out_dir, "table4_run_counts.csv")
    table4.to_csv(csv_path, index=False)

    # LaTeX output
    tex_path = os.path.join(out_dir, "table4_run_counts.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Table 4: Completed Run Counts per Policy and Evaluation Graph\n")
        cols_fmt = "l" + "c" * (len(actual_cols) - 1)
        f.write(f"\\begin{{tabular}}{{{cols_fmt}}}\n")
        f.write("\\hline\n")
        f.write(" & ".join(actual_cols) + " \\\\\n")
        f.write("\\hline\n")
        for _, r in table4.iterrows():
            f.write(" & ".join([str(r[c]) for c in actual_cols]) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    return table4


def main():
    out_dir = "results/tables"
    report_dir = "report/tables"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    df, meta = load_and_validate()
    run_totals = compute_run_total_times(df)
    stats_df = compute_policy_graph_stats(run_totals, seed=12345)
    overall_pass, pf_df = run_pass_fail_check(stats_df)

    t1 = generate_table1_graph_properties(out_dir=out_dir)
    t2 = generate_table2_policy_times(stats_df, out_dir=out_dir)
    t3 = generate_table3_pass_fail(pf_df, out_dir=out_dir)
    t4 = generate_table4_run_counts(run_totals, out_dir=out_dir)

    # Copy files to report/tables
    for name in ["table1_graph_properties", "table2_policy_times", "table3_pass_fail", "table4_run_counts"]:
        for ext in [".csv", ".tex"]:
            src = os.path.join(out_dir, name + ext)
            dst = os.path.join(report_dir, name + ext)
            if os.path.exists(src):
                with open(src, "r", encoding="utf-8") as f_in, open(dst, "w", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())

    print(f"[tables.py] Successfully generated Tables 1 to 4 in {out_dir} and {report_dir}")


if __name__ == "__main__":
    main()
