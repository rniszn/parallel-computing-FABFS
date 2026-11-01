#!/usr/bin/env python3
"""
M10: Fallback screenshot generator.
Renders static fallback images for each step of the demo run into presentation/fallback/,
ensuring the presenter has a complete backup in case of live browser presentation issues.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo")))
from build_demo import load_binary_graph, generate_node_layout, validate_demo_run


def generate_fallback_images(out_dir="presentation/fallback"):
    os.makedirs(out_dir, exist_ok=True)
    demo_df, run_id, graph_name = validate_demo_run()
    n, m, row_offsets, col_indices = load_binary_graph("demo/demo_graph.bin")
    coords, edges = generate_node_layout(n, row_offsets, col_indices, demo_df)

    with open("config/thresholds.json", "r", encoding="utf-8") as f:
        thresh = json.load(f)

    # Coords arrays
    xs = np.array([c["x"] for c in coords])
    ys = np.array([c["y"] for c in coords])
    node_levels = np.array([c["level"] for c in coords])

    num_levels = len(demo_df)

    mode_colors = {
        "SEQ": "#38bdf8",
        "STATIC": "#4ade80",
        "DYNAMIC": "#fb923c"
    }

    for lvl in range(num_levels):
        r = demo_df.iloc[lvl]
        mode = r["mode_chosen"]
        frontier_size = int(r["frontier_size"])
        time_us = float(r["time_us"])
        cv = float(r["cv"])

        fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="#0f172a")
        ax.set_facecolor("#090d16")

        # Draw sampled edges
        for e in edges[:500]:
            u, v = e["u"], e["v"]
            ax.plot([coords[u]["x"], coords[v]["x"]], [coords[u]["y"], coords[v]["y"]],
                    color="#334155", alpha=0.3, linewidth=0.5, zorder=1)

        # Draw unvisited nodes
        unvis = (node_levels > lvl) & (np.arange(n) != 0)
        ax.scatter(xs[unvis], ys[unvis], s=8, color="#334155", alpha=0.7, zorder=2)

        # Draw past visited nodes
        if lvl > 0:
            past = (node_levels < lvl) & (np.arange(n) != 0)
            ax.scatter(xs[past], ys[past], s=16, color="#64748b", alpha=0.85, zorder=3)

        # Draw active frontier nodes
        curr = (node_levels == lvl) & (np.arange(n) != 0)
        c_mode = mode_colors.get(mode, "#38bdf8")
        if np.any(curr):
            ax.scatter(xs[curr], ys[curr], s=40, color=c_mode, edgecolors="white", linewidth=0.8, zorder=5, label=f"Active Frontier ({mode})")

        # Source node (0)
        ax.scatter([xs[0]], [ys[0]], s=80, color="#f59e0b", edgecolors="white", linewidth=1.5, zorder=6, label="Source Vertex")

        # Telemetry Card
        hud_text = (
            f"Level {lvl} / {num_levels-1} | Mode: {mode}\n"
            f"Frontier Size: {frontier_size:,} | Skew CV: {cv:.2f}\n"
            f"Level Time: {time_us:.1f} us\n"
            f"Decision Rule: {'|F| < T_min' if mode=='SEQ' else ('CV <= T_CV' if mode=='STATIC' else 'CV > T_CV & |F| > T_par')}\n"
            f"Commit: {thresh.get('calibration_commit', '')[:8]}"
        )
        ax.text(0.03, 0.95, hud_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', color="white", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#1e293b", edgecolor=c_mode, linewidth=1.5))

        ax.set_title(f"FABFS Demo Replay — Step {lvl} (Level {lvl})", color="white", fontsize=13, fontweight="bold", pad=12)
        ax.axis("off")
        fig.tight_layout()

        out_img = os.path.join(out_dir, f"step_level_{lvl}.png")
        fig.savefig(out_img, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        print(f"[generate_fallback] Wrote {out_img}")

    print(f"[generate_fallback] All {num_levels} fallback images generated in {out_dir}")


if __name__ == "__main__":
    generate_fallback_images()
