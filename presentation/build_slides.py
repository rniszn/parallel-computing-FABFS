#!/usr/bin/env python3
"""
M10: Slide Deck Builder.
Copies figures from M9 analysis/report into presentation/figures/,
writes presentation/figures/checksums.txt,
and generates the complete presentation/FABFS.pptx deck using python-pptx.
Every slide quoting a number includes an explicit footnote to its M9 table/figure source.
"""

import os
import sys
import shutil
import hashlib
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

FIGURE_FILES = [
    "fig1_median_times.png",
    "fig2_rmat_levels.png",
    "fig3_mode_shares.png",
    "fig4_cpu_atomics.png",
    "fig5_gpu_proxy_idle.png"
]


def copy_figures_and_checksums(src_dir="report/figures", dst_dir="presentation/figures"):
    """
    Copies M9 figure outputs into presentation/figures/ and generates checksums.txt.
    """
    os.makedirs(dst_dir, exist_ok=True)
    checksum_lines = []

    for fname in FIGURE_FILES:
        src = os.path.join(src_dir, fname)
        if not os.path.exists(src):
            src = os.path.join("results/figures", fname)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Required figure {fname} not found in {src_dir} or results/figures")

        dst = os.path.join(dst_dir, fname)
        shutil.copy2(src, dst)

        # Compute SHA256
        with open(dst, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        checksum_lines.append(f"{h}  {fname}\n")
        print(f"[build_slides.py] Copied {fname} -> {dst} (SHA256: {h[:12]}...)")

    chk_path = os.path.join(dst_dir, "checksums.txt")
    with open(chk_path, "w", encoding="utf-8") as f:
        f.writelines(checksum_lines)
    print(f"[build_slides.py] Written figure checksums to {chk_path}")


def add_slide_header(slide, title_text, category="FABFS | IT-301 PARALLEL COMPUTING"):
    """Helper to add consistent slide headers."""
    # Category tag
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.3))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category.upper()
    p_c.font.size = Pt(10)
    p_c.font.bold = True
    p_c.font.color.rgb = RGBColor(2, 132, 199) # Blue

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(24)
    p_t.font.bold = True
    p_t.font.color.rgb = RGBColor(15, 23, 42)


def add_footnote(slide, footnote_text):
    """Helper to add strict provenance footnotes naming M9 tables/figures."""
    fn_box = slide.shapes.add_textbox(Inches(0.8), Inches(6.9), Inches(11.7), Inches(0.4))
    tf = fn_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"* Source provenance: {footnote_text}"
    p.font.size = Pt(9)
    p.font.italic = True
    p.font.color.rgb = RGBColor(100, 116, 139)


def build_deck(out_pptx_path="presentation/FABFS.pptx"):
    """
    Constructs the 12-slide presentation deck.
    """
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # ----------------------------------------------------
    # SLIDE 1: Title Slide
    # ----------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.0))
    tf = t_box.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "NITK SURATHKAL — IT-301 PARALLEL COMPUTING"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(2, 132, 199)

    p1 = tf.add_paragraph()
    p1.text = "FABFS: Frontier-Adaptive CPU-GPU BFS"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(15, 23, 42)
    p1.space_before = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "Dynamic Phase Transitions and Work-Stealing on Unified Managed Memory"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(71, 85, 105)
    p2.space_before = Pt(8)

    meta_box = s1.shapes.add_textbox(Inches(1.0), Inches(5.2), Inches(11.3), Inches(1.5))
    tf_m = meta_box.text_frame
    p_m1 = tf_m.paragraphs[0]
    p_m1.text = "Author / Candidate: Rohan G (NITK) | Advisor / Evaluator Review"
    p_m1.font.size = Pt(12)
    p_m1.font.bold = True
    p_m1.font.color.rgb = RGBColor(30, 41, 59)

    p_m2 = tf_m.add_paragraph()
    p_m2.text = "Target Architecture: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, Ada Lovelace sm_89) + 13th Gen Intel Core i5-13420H"
    p_m2.font.size = Pt(11)
    p_m2.font.color.rgb = RGBColor(100, 116, 139)

    # ----------------------------------------------------
    # SLIDE 2: Problem & Heterogeneous Frontier Imbalance
    # ----------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_header(s2, "The Challenge: Frontier Phase Changes & Work Imbalance")

    b_box = s2.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.0))
    tf = b_box.text_frame
    tf.word_wrap = True

    items = [
        ("Extreme Scale Variations Across BFS Levels:",
         "Breadth-First Search traversals exhibit orders-of-magnitude frontier expansion. Boundary levels (root and leaf tiers) contain tens of vertices, while mid-traversal levels surge to hundreds of thousands of active nodes."),
        ("Static Partitioning Bottlenecks:",
         "Offloading small frontiers to GPU wastes significant time in kernel launch overhead (~40 us) and memory synchronization. Furthermore, static edge-splitting stalls on hub-heavy power-law distributions due to intra-warp load imbalance."),
        ("Pure Dynamic Work-Stealing Pitfalls:",
         "While Chase-Lev work-stealing balances skewed frontiers, running fine-grained dynamic queues on uniform, dense graphs induces unnecessary atomic CAS contention on Unified Managed Memory."),
        ("The Frontier-Adaptive Approach (FABFS):",
         "Online lightweight characterization of frontier size (|F_l|) and degree skew (CV) to dynamically transition between Sequential CPU, Static Split, and Dynamic Work-Stealing without manual tuning.")
    ]
    for idx, (h, desc) in enumerate(items):
        p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
        p.text = f"{idx+1}. {h}"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = RGBColor(15, 23, 42)
        p.space_before = Pt(10)

        p_sub = tf.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = RGBColor(71, 85, 105)
        p_sub.space_before = Pt(3)

    add_footnote(s2, "Problem statement & architectural trade-offs defined in Project_Plan.pdf and M6.pdf")

    # ----------------------------------------------------
    # SLIDE 3: Decision Engine & Execution Modes
    # ----------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_header(s3, "Algorithm 1: Frontier-Adaptive Decision Engine")

    b_box = s3.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(4.8))
    tf = b_box.text_frame
    tf.word_wrap = True

    modes = [
        ("Rule 1: Sequential CPU Mode (SEQ)",
         "Condition: |F_l| < T_min (1,024 vertices)\n"
         "Mechanism: Dispatched strictly to single CPU core. Completely bypasses thread pool synchronization and GPU kernel launch latency."),
        ("Rule 2: Static Work Partitioning (STATIC)",
         "Condition: CV <= T_CV (1.5) OR |F_l| <= T_par (8,192 vertices)\n"
         "Mechanism: Prefix sum degree search splits frontier by edge count. GPU expands alpha = 15.26% of edge work; CPU OpenMP threads process remaining work in parallel."),
        ("Rule 3: Dynamic Work-Stealing (DYNAMIC)",
         "Condition: High degree skew CV > 1.5 AND |F_l| > 8,192 vertices\n"
         "Mechanism: Lock-free Chase-Lev deques with weak memory orders. CPU workers execute local LIFO tasks while a dedicated GPU proxy steals and coalesces batches >= 4,096 vertices.")
    ]
    for idx, (h, desc) in enumerate(modes):
        p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
        p.text = f"• {h}"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = RGBColor(2, 132, 199) if "SEQ" in h else (RGBColor(22, 163, 74) if "STATIC" in h else RGBColor(234, 88, 12))
        p.space_before = Pt(12)

        p_sub = tf.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = RGBColor(51, 65, 85)
        p_sub.space_before = Pt(3)

    add_footnote(s3, "Frozen calibrated parameters: config/thresholds.json (commit: 8dd8ad730e2eb6c5bf6e9491eb8401f9c7d792e3)")

    # ----------------------------------------------------
    # SLIDE 4: Evaluator Demo: Real Log Replay
    # ----------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_header(s4, "Evaluator Demo: Real Log Replay Visualizer")

    b_box = s4.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(4.8))
    tf = b_box.text_frame
    tf.word_wrap = True

    demo_bullets = [
        ("Replay from Real Execution Telemetry (Not a Mock-up):",
         "The visualizer (demo/fabfs_demo.html) directly loads level rows produced by the M7 logger in results/levels.csv. It refuses to build on any hash or mode mismatch."),
        ("Interactive Canvas & Concentric Layout:",
         "Vertices are laid out deterministically. Color-coded states reflect level progression and expanding hardware device (CPU blue, GPU green, Dynamic orange)."),
        ("Live Decision HUD & Stealing Telemetry:",
         "Provides real-time inspection of |F_l|, active edges, degree variance (CV), chunk sizes (C_l), steal success rates, and GPU proxy idle times."),
        ("Self-Contained & Completely Offline:",
         "Single HTML file requiring zero network requests, zero CDN links, and zero runtime dependencies.")
    ]
    for idx, (h, desc) in enumerate(demo_bullets):
        p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
        p.text = f"✔ {h}"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = RGBColor(15, 23, 42)
        p.space_before = Pt(10)

        p_sub = tf.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = RGBColor(71, 85, 105)
        p_sub.space_before = Pt(3)

    add_footnote(s4, "Verified run: demo_rmat10_run1 (demo/demo_run.csv, demo/demo_graph.bin)")

    # ----------------------------------------------------
    # SLIDE 5: Full Benchmark Execution Matrix
    # ----------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_header(s5, "Benchmark Execution Matrix (480 Total Runs)")

    b_box = s5.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(4.8))
    tf = b_box.text_frame
    tf.word_wrap = True

    matrix_info = [
        ("Matrix Dimensions: 4 Policies x 8 Evaluation Graphs x 15 Rounds = 480 Runs",
         "Includes 2 unrecorded warm-up runs per cell to eliminate cold-start caching and page-fault artifacts."),
        ("Strict Topology Isolation:",
         "Calibration graphs (roadNet-PA, com-Amazon, com-DBLP, rmat_s16) strictly forbidden from evaluation matrix. Verified by runner manifest check."),
        ("Evaluation Topologies Represented:",
         "• Planar Road Networks: roadNet-CA (1.97M nodes, 5.53M edges)\n"
         "• Web & Social Networks: web-Google (875K nodes, 8.64M edges), com-LiveJournal (3.99M nodes, 69.36M edges), com-Orkut (3.07M nodes, 234.37M edges)\n"
         "• Synthetic R-MAT Topologies: Scale 20, 1M nodes, a in {0.25, 0.45, 0.55, 0.65}"),
        ("Resilient Crash-Recovery Architecture:",
         "Runner logs completions to results/runs.jsonl with atomic fsync. Any interrupted run has partial rows removed and restarts cleanly.")
    ]
    for idx, (h, desc) in enumerate(matrix_info):
        p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
        p.text = f"• {h}"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = RGBColor(15, 23, 42)
        p.space_before = Pt(8)

        p_sub = tf.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(11.5)
        p_sub.font.color.rgb = RGBColor(71, 85, 105)
        p_sub.space_before = Pt(2)

    add_footnote(s5, "Table 1 (results/tables/table1_graph_properties.csv) and Table 4 (results/tables/table4_run_counts.csv)")

    # ----------------------------------------------------
    # SLIDE 6: Traversal Performance Across Policies (Fig 1)
    # ----------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_header(s6, "Traversal Performance: Median Total Time by Policy")
    fig1_path = "presentation/figures/fig1_median_times.png"
    if os.path.exists(fig1_path):
        s6.shapes.add_picture(fig1_path, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.0))
    add_footnote(s6, "Figure 1 (report/figures/fig1_median_times.png) & Table 2 (report/tables/table2_policy_times.csv). 95% bootstrap CI.")

    # ----------------------------------------------------
    # SLIDE 7: Per-Level Mode Transitions (Fig 2)
    # ----------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_header(s7, "Adaptive Mode Transitions: Level Execution Trace")
    fig2_path = "presentation/figures/fig2_rmat_levels.png"
    if os.path.exists(fig2_path):
        s7.shapes.add_picture(fig2_path, Inches(1.8), Inches(1.6), Inches(9.7), Inches(5.0))
    add_footnote(s7, "Figure 2 (report/figures/fig2_rmat_levels.png). Shaded bands reflect SEQ (blue), STATIC (green), DYNAMIC (orange).")

    # ----------------------------------------------------
    # SLIDE 8: Mode Selection Distribution (Fig 3)
    # ----------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_header(s8, "Dispatch Mode Distribution Across Topologies")
    fig3_path = "presentation/figures/fig3_mode_shares.png"
    if os.path.exists(fig3_path):
        s8.shapes.add_picture(fig3_path, Inches(1.8), Inches(1.6), Inches(9.7), Inches(5.0))
    add_footnote(s8, "Figure 3 (report/figures/fig3_mode_shares.png). Share of BFS levels spent in SEQ, STATIC, and DYNAMIC modes.")

    # ----------------------------------------------------
    # SLIDE 9: Cost of Shared Atomics & Managed Memory (Fig 4)
    # ----------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_header(s9, "Overhead Analysis: CPU Thread Scaling & Managed Memory Atomics")
    fig4_path = "presentation/figures/fig4_cpu_atomics.png"
    if os.path.exists(fig4_path):
        s9.shapes.add_picture(fig4_path, Inches(2.0), Inches(1.6), Inches(9.3), Inches(5.0))
    add_footnote(s9, "Figure 4 (report/figures/fig4_cpu_atomics.png) and Gate 1 evaluation telemetry (results/gate1.txt)")

    # ----------------------------------------------------
    # SLIDE 10: GPU Proxy Idle Overhead (Fig 5)
    # ----------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_header(s10, "Telemetry: GPU Proxy Idle Overhead per Level")
    fig5_path = "presentation/figures/fig5_gpu_proxy_idle.png"
    if os.path.exists(fig5_path):
        s10.shapes.add_picture(fig5_path, Inches(2.0), Inches(1.6), Inches(9.3), Inches(5.0))
    add_footnote(s10, "Figure 5 (report/figures/fig5_gpu_proxy_idle.png). Distribution of gpu_idle_us across evaluation graphs.")

    # ----------------------------------------------------
    # SLIDE 11: Acceptance Criteria: Formal Pass/Fail
    # ----------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    add_slide_header(s11, "Acceptance Verification: Adaptive Policy Pass/Fail Check")

    b_box = s11.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(4.8))
    tf = b_box.text_frame
    tf.word_wrap = True

    pf_info = [
        ("The Strict 5% Evaluation Rule:",
         "Criterion: On every evaluation graph, the adaptive median total time must be within 5% of the better of Pure-Static and Pure-Dynamic:\n"
         "T_adaptive <= 1.05 * min(T_static, T_dynamic). Zero manual overrides allowed."),
        ("Empirical Findings on Evaluated Topologies:",
         "• Road Networks (roadNet-CA): Adaptive achieves 1,277 ms vs Pure-Static 2,771 ms (53.9% faster, clear PASS).\n"
         "• Power-Law Web Graph (web-Google): Adaptive achieves 1,593 ms vs Pure-Dynamic 1,751 ms (10% speedup).\n"
         "• Synthetic R-MAT Graphs: Dynamic work-stealing mitigates extreme degree skew (CV > 3.5), maintaining near-oracle parity."),
        ("Deterministic Gate Verification:",
         "Pass/fail logic is codified in analysis/stats.py and automatically checked via tests/test_analysis.py. Scripts exit with non-zero status on threshold violation.")
    ]
    for idx, (h, desc) in enumerate(pf_info):
        p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
        p.text = f"✔ {h}"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = RGBColor(15, 23, 42)
        p.space_before = Pt(10)

        p_sub = tf.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = RGBColor(71, 85, 105)
        p_sub.space_before = Pt(3)

    add_footnote(s11, "Table 3 (report/tables/table3_pass_fail.csv) and analysis/stats.py")

    # ----------------------------------------------------
    # SLIDE 12: Summary of Deliverables
    # ----------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    add_slide_header(s12, "Summary of Deliverables & Project Completion")

    b_box = s12.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(4.8))
    tf = b_box.text_frame
    tf.word_wrap = True

    summary_items = [
        ("Core Engineering Modules (M0 - M8):",
         "Verified CUDA level-synchronous kernel (M4), OpenMP CPU expander (M3), lock-free Chase-Lev deques (M5), cost model & decision engine (M6), 28-column logger (M7), and crash-resilient matrix runner (M8)."),
        ("Reproducible Analysis Suite (M9):",
         "Automated data loader, bootstrap confidence intervals, LaTeX tables, high-resolution figures, and official report provenance numbers generated from raw CSV."),
        ("Live Demonstration & Visual Presentation (M10):",
         "Interactive replay visualizer (demo/fabfs_demo.html), complete 16:9 presentation deck (presentation/FABFS.pptx), and presenter walk-through guide (presentation/WALKTHROUGH.md)."),
        ("Git & Verification Integrity:",
         "Every module developed on dedicated branches with clean PRs; all gates and test suites pass.")
    ]
    for idx, (h, desc) in enumerate(summary_items):
        p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
        p.text = f"★ {h}"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = RGBColor(2, 132, 199)
        p.space_before = Pt(10)

        p_sub = tf.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = RGBColor(71, 85, 105)
        p_sub.space_before = Pt(3)

    add_footnote(s12, "Project: FABFS (Frontier-Adaptive CPU-GPU BFS), IT-301 Parallel Computing, NITK")

    prs.save(out_pptx_path)
    print(f"[build_slides.py] Successfully created PowerPoint presentation: {out_pptx_path}")


def main():
    print("[build_slides.py] Copying M9 figures and building slide deck...")
    copy_figures_and_checksums()
    build_deck()
    print("[build_slides.py] Presentation build complete.")


if __name__ == "__main__":
    main()
