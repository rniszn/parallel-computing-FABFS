#!/usr/bin/env python3
"""
M9: Figures generator.
Generates Figures 1 to 5 as vector PDF and high-res PNG, with corresponding CSV data files.
Uses a unified, professional color palette across all figures without manual hand-tuning.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load import load_and_validate
from stats import compute_run_total_times, compute_policy_graph_stats

# Consistent publication color palette
PALETTE = {
    "Adaptive": "#2b5c8f",      # Deep blue
    "Pure-Static": "#2ca02c",   # Green
    "Pure-Dynamic": "#d65f5f",  # Coral/Red
    "Oracle-Replay": "#9467bd", # Purple
    "SEQ": "#4C72B0",
    "STATIC": "#55A868",
    "DYNAMIC": "#DD8452",
    "bg_seq": "#d0e1fd",
    "bg_static": "#d5f5e3",
    "bg_dynamic": "#fdebd0"
}

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})


def plot_fig1_median_times(stats_df, out_dir="results/figures"):
    """
    Figure 1: Median total time per policy, one panel per graph, with CI bars.
    """
    graphs = sorted(stats_df["graph"].unique())
    policies = ["Adaptive", "Pure-Static", "Pure-Dynamic", "Oracle-Replay"]

    n_graphs = len(graphs)
    cols = 4 if n_graphs >= 4 else n_graphs
    rows = (n_graphs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.2 * rows), squeeze=False)
    fig.suptitle("Figure 1: Median Traversal Time by Policy with 95% Bootstrap CI", y=1.02)

    data_rows = []

    for idx, g in enumerate(graphs):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        sub = stats_df[stats_df["graph"] == g].set_index("policy")
        x_pos = np.arange(len(policies))
        heights = []
        err_lower = []
        err_upper = []
        bar_colors = []

        for p in policies:
            if p in sub.index:
                med = sub.loc[p, "median_ms"]
                lo = sub.loc[p, "ci_lower_ms"]
                hi = sub.loc[p, "ci_upper_ms"]
                heights.append(med)
                err_lower.append(max(0.0, med - lo))
                err_upper.append(max(0.0, hi - med))
                bar_colors.append(PALETTE.get(p, "#333333"))
                data_rows.append({
                    "graph": g, "policy": p,
                    "median_ms": med, "ci_lower_ms": lo, "ci_upper_ms": hi
                })
            else:
                heights.append(0)
                err_lower.append(0)
                err_upper.append(0)
                bar_colors.append("#cccccc")

        yerr = [err_lower, err_upper]
        ax.bar(x_pos, heights, yerr=yerr, capsize=4, color=bar_colors, edgecolor="black", linewidth=0.7)
        ax.set_title(g, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Adapt", "Static", "Dynamic", "Oracle"], rotation=25)
        ax.set_ylabel("Time (ms)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Hide unused subplots
    for idx in range(n_graphs, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r, c])

    fig.tight_layout()

    pdf_path = os.path.join(out_dir, "fig1_median_times.pdf")
    png_path = os.path.join(out_dir, "fig1_median_times.png")
    csv_path = os.path.join(out_dir, "fig1_median_times.csv")

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    pd.DataFrame(data_rows).to_csv(csv_path, index=False)
    return pdf_path, png_path, csv_path


def plot_fig2_rmat_levels(df, out_dir="results/figures"):
    """
    Figure 2: Per-level time for one R-MAT graph, with the mode chosen for each level shaded.
    """
    # Prefer rmat_a0.55 or any available R-MAT graph with Adaptive policy
    rmat_graphs = [g for g in df["graph"].unique() if "rmat" in g]
    target_graph = "rmat_a0.55" if "rmat_a0.55" in rmat_graphs else (rmat_graphs[0] if rmat_graphs else df["graph"].iloc[0])

    sub = df[(df["graph"] == target_graph) & (df["policy"] == "Adaptive")]
    if sub.empty:
        sub = df[df["graph"] == target_graph]

    # Pick representative round (median round or lowest round)
    sample_round = sub["round"].min()
    run_df = sub[sub["round"] == sample_round].sort_values("level").copy()

    levels = run_df["level"].values
    times_ms = run_df["time_us"].values / 1000.0
    modes = run_df["mode_chosen"].values

    fig, ax = plt.subplots(figsize=(7, 4))

    # Shading for modes
    mode_color_map = {
        "SEQ": PALETTE["bg_seq"],
        "STATIC": PALETTE["bg_static"],
        "DYNAMIC": PALETTE["bg_dynamic"]
    }

    for i in range(len(levels)):
        lvl = levels[i]
        m = modes[i]
        ax.axvspan(lvl - 0.45, lvl + 0.45, color=mode_color_map.get(m, "#eeeeee"), alpha=0.6, lw=0)

    # Plot time curve
    ax.plot(levels, times_ms, marker="o", color="#1f4e78", linewidth=1.8, markersize=5, label="Level Execution Time")
    ax.set_xlabel("BFS Level")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title(f"Figure 2: Per-Level Execution Time & Chosen Dispatch Mode ({target_graph})", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xticks(levels)

    # Mode legend patches
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=mode_color_map["SEQ"], label="Mode: SEQ"),
        mpatches.Patch(color=mode_color_map["STATIC"], label="Mode: STATIC"),
        mpatches.Patch(color=mode_color_map["DYNAMIC"], label="Mode: DYNAMIC"),
    ]
    ax.legend(handles=[ax.lines[0]] + patches, loc="upper right")

    fig.tight_layout()

    pdf_path = os.path.join(out_dir, "fig2_rmat_levels.pdf")
    png_path = os.path.join(out_dir, "fig2_rmat_levels.png")
    csv_path = os.path.join(out_dir, "fig2_rmat_levels.csv")

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    run_df[["level", "frontier_size", "edge_work", "cv", "mode_chosen", "time_us"]].to_csv(csv_path, index=False)
    return pdf_path, png_path, csv_path


def plot_fig3_mode_shares(df, out_dir="results/figures"):
    """
    Figure 3: Share of levels in each mode, per graph and policy.
    """
    graphs = sorted(df["graph"].unique())
    # Group by graph, compute proportion of levels in SEQ, STATIC, DYNAMIC for Adaptive policy
    adapt_df = df[df["policy"] == "Adaptive"]

    data_rows = []
    for g in graphs:
        gdf = adapt_df[adapt_df["graph"] == g]
        total = len(gdf)
        if total > 0:
            seq_share = (gdf["mode_chosen"] == "SEQ").sum() / total * 100.0
            stat_share = (gdf["mode_chosen"] == "STATIC").sum() / total * 100.0
            dyn_share = (gdf["mode_chosen"] == "DYNAMIC").sum() / total * 100.0
        else:
            seq_share, stat_share, dyn_share = 0, 0, 0
        data_rows.append({
            "graph": g,
            "SEQ_pct": seq_share,
            "STATIC_pct": stat_share,
            "DYNAMIC_pct": dyn_share
        })

    mdf = pd.DataFrame(data_rows)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x_pos = np.arange(len(graphs))
    bar_width = 0.55

    ax.bar(x_pos, mdf["SEQ_pct"], width=bar_width, label="SEQ", color=PALETTE["SEQ"], edgecolor="black", linewidth=0.5)
    ax.bar(x_pos, mdf["STATIC_pct"], width=bar_width, bottom=mdf["SEQ_pct"], label="STATIC", color=PALETTE["STATIC"], edgecolor="black", linewidth=0.5)
    ax.bar(x_pos, mdf["DYNAMIC_pct"], width=bar_width, bottom=mdf["SEQ_pct"] + mdf["STATIC_pct"], label="DYNAMIC", color=PALETTE["DYNAMIC"], edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Share of Traversal Levels (%)")
    ax.set_title("Figure 3: Distribution of Adaptive Dispatch Modes per Graph", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(graphs, rotation=25, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()

    pdf_path = os.path.join(out_dir, "fig3_mode_shares.pdf")
    png_path = os.path.join(out_dir, "fig3_mode_shares.png")
    csv_path = os.path.join(out_dir, "fig3_mode_shares.csv")

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    mdf.to_csv(csv_path, index=False)
    return pdf_path, png_path, csv_path


def plot_fig4_cpu_atomics(gate1_path="results/gate1.txt", out_dir="results/figures"):
    """
    Figure 4: Cost of CPU atomics on managed memory from M5 timing run.
    Parses results/gate1.txt to compare thread scaling across modes.
    """
    if not os.path.exists(gate1_path):
        raise FileNotFoundError(f"{gate1_path} not found")

    records = []
    with open(gate1_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) == 5:
                # graph,mode,threads,time_ms,correct
                try:
                    records.append({
                        "graph": parts[0],
                        "mode": parts[1],
                        "threads": int(parts[2]),
                        "time_ms": float(parts[3]),
                        "correct": parts[4] == "true"
                    })
                except ValueError:
                    continue

    gdf = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "fig4_cpu_atomics.csv")
    gdf.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 4.2))

    # Average normalized time relative to 1 thread for SEQ, STATIC, DYNAMIC
    modes = ["SEQ", "STATIC", "DYNAMIC", "ADAPTIVE"]
    threads_list = [1, 2, 4, 8]

    for m in modes:
        sub = gdf[gdf["mode"] == m]
        if sub.empty:
            continue
        avg_times = []
        for t in threads_list:
            t_sub = sub[sub["threads"] == t]
            avg_times.append(t_sub["time_ms"].mean() if not t_sub.empty else np.nan)
        ax.plot(threads_list, avg_times, marker="s", label=f"Mode: {m}", linewidth=1.8, markersize=5)

    ax.set_xlabel("CPU OpenMP Threads")
    ax.set_ylabel("Mean Traversal Time (ms)")
    ax.set_title("Figure 4: CPU Scaling & Atomic Contention on Managed State (M5 Gate 1)", fontweight="bold")
    ax.set_xticks(threads_list)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    fig.tight_layout()

    pdf_path = os.path.join(out_dir, "fig4_cpu_atomics.pdf")
    png_path = os.path.join(out_dir, "fig4_cpu_atomics.png")

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    return pdf_path, png_path, csv_path


def plot_fig5_gpu_proxy_idle(df, out_dir="results/figures"):
    """
    Figure 5: GPU proxy idle time per level (gpu_idle_us).
    """
    fig, ax = plt.subplots(figsize=(8, 4.2))

    # Filter for runs that record gpu_idle_us > 0
    sub = df[df["gpu_idle_us"] >= 0]
    graphs = sorted(sub["graph"].unique())

    box_data = []
    labels = []
    summary_data = []

    for g in graphs:
        vals = sub[sub["graph"] == g]["gpu_idle_us"].values
        if len(vals) > 0:
            box_data.append(vals)
            labels.append(g)
            summary_data.append({
                "graph": g,
                "median_gpu_idle_us": np.median(vals),
                "mean_gpu_idle_us": np.mean(vals),
                "max_gpu_idle_us": np.max(vals),
                "p95_gpu_idle_us": np.percentile(vals, 95)
            })

    if box_data:
        bp = ax.boxplot(box_data, patch_artist=True, labels=labels, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor("#a1c9f4")
            patch.set_edgecolor("#1f4e78")
        for median in bp['medians']:
            median.set_color("#8c0000")
            median.set_linewidth(1.5)

    ax.set_ylabel("GPU Proxy Idle Time (us)")
    ax.set_title("Figure 5: GPU Proxy Idle Overhead per Level Across Evaluation Graphs", fontweight="bold")
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()

    pdf_path = os.path.join(out_dir, "fig5_gpu_proxy_idle.pdf")
    png_path = os.path.join(out_dir, "fig5_gpu_proxy_idle.png")
    csv_path = os.path.join(out_dir, "fig5_gpu_proxy_idle.csv")

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    pd.DataFrame(summary_data).to_csv(csv_path, index=False)
    return pdf_path, png_path, csv_path


def main():
    out_dir = "results/figures"
    report_dir = "report/figures"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    df, meta = load_and_validate()
    run_totals = compute_run_total_times(df)
    stats_df = compute_policy_graph_stats(run_totals, seed=12345)

    plot_fig1_median_times(stats_df, out_dir=out_dir)
    plot_fig2_rmat_levels(df, out_dir=out_dir)
    plot_fig3_mode_shares(df, out_dir=out_dir)
    plot_fig4_cpu_atomics(out_dir=out_dir)
    plot_fig5_gpu_proxy_idle(df, out_dir=out_dir)

    # Copy generated figures and data to report/figures/
    for name in ["fig1_median_times", "fig2_rmat_levels", "fig3_mode_shares", "fig4_cpu_atomics", "fig5_gpu_proxy_idle"]:
        for ext in [".pdf", ".png", ".csv"]:
            src = os.path.join(out_dir, name + ext)
            dst = os.path.join(report_dir, name + ext)
            if os.path.exists(src):
                with open(src, "rb") as f_in, open(dst, "wb") as f_out:
                    f_out.write(f_in.read())

    print(f"[figures.py] Successfully generated Figures 1 to 5 in {out_dir} and {report_dir}")


if __name__ == "__main__":
    main()
