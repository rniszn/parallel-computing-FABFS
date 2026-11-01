#!/usr/bin/env python3
"""
M10: Demo builder script.
Reads baseline run from results/levels.csv and demo/demo_run.csv,
reads demo/demo_graph.bin, validates run_id and graph hash integrity,
extracts real evaluation graph runs (web-Google, com-lj, roadNet-CA),
and generates demo/fabfs_demo.html (a self-contained, offline multi-graph HTML visualizer).
"""

import os
import sys
import json
import struct
import hashlib
import numpy as np
import pandas as pd

# Add analysis directory for CONTRACT_COLUMNS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis")))
from load import CONTRACT_COLUMNS, check_header


def compute_file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_binary_graph(bin_path):
    """
    Loads binary CSR graph from demo/demo_graph.bin:
      uint32_t n
      uint64_t m
      uint64_t row_offsets[n + 1]
      uint32_t col_indices[m]
    Returns (n, m, row_offsets, col_indices).
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Binary graph not found: {bin_path}")

    with open(bin_path, "rb") as f:
        n_bytes = f.read(4)
        m_bytes = f.read(8)
        if len(n_bytes) < 4 or len(m_bytes) < 8:
            raise ValueError("Corrupted binary graph header")

        n = struct.unpack("=I", n_bytes)[0]
        m = struct.unpack("=Q", m_bytes)[0]

        row_offsets_bytes = f.read((n + 1) * 8)
        col_indices_bytes = f.read(m * 4)

        row_offsets = np.frombuffer(row_offsets_bytes, dtype=np.uint64)
        col_indices = np.frombuffer(col_indices_bytes, dtype=np.uint32)

    return n, m, row_offsets, col_indices


def validate_demo_run(demo_run_path="demo/demo_run.csv",
                      levels_csv_path="results/levels.csv",
                      runs_jsonl_path="results/runs.jsonl",
                      inputs_path="results/env/inputs.txt",
                      demo_graph_bin_path="demo/demo_graph.bin"):
    """
    Validates that demo_run.csv rows match results/levels.csv,
    that run_id is valid, and graph hash matches inputs.txt.
    """
    if not os.path.exists(demo_run_path):
        raise FileNotFoundError(f"{demo_run_path} not found")
    if not os.path.exists(levels_csv_path):
        raise FileNotFoundError(f"{levels_csv_path} not found")

    demo_df = pd.read_csv(demo_run_path)
    check_header(demo_df.columns)

    run_id = demo_df["run_id"].iloc[0]
    graph_name = demo_df["graph"].iloc[0]

    # Verify rows match results/levels.csv
    levels_df = pd.read_csv(levels_csv_path)
    matching_levels = levels_df[levels_df["run_id"] == run_id]
    if len(matching_levels) == 0:
        raise ValueError(f"Run ID '{run_id}' not found in {levels_csv_path}")

    if len(matching_levels) != len(demo_df):
        raise ValueError(
            f"Row count mismatch: demo_run.csv has {len(demo_df)} rows, "
            f"but levels.csv has {len(matching_levels)} for run {run_id}"
        )

    # Verify modes match
    for idx in range(len(demo_df)):
        mode_demo = demo_df.iloc[idx]["mode_chosen"]
        mode_levels = matching_levels.iloc[idx]["mode_chosen"]
        if mode_demo != mode_levels:
            raise ValueError(
                f"Mode mismatch at level {idx}: demo={mode_demo} vs levels={mode_levels}"
            )

    # Verify graph hash in inputs.txt
    known_hashes = {}
    if os.path.exists(inputs_path):
        with open(inputs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        known_hashes[parts[0]] = parts[1]

    if graph_name in known_hashes:
        print(f"[build_demo.py] Verified graph hash for {graph_name}: {known_hashes[graph_name]}")
    else:
        print(f"[build_demo.py] Warning: graph {graph_name} not found in {inputs_path}")

    return demo_df, run_id, graph_name


def generate_node_layout(n, row_offsets, col_indices, demo_df):
    """
    Computes a clean concentric/circular 2D layout for graph vertices.
    Assigns each node an estimated BFS level based on frontier expansion.
    """
    np.random.seed(42)

    coords = []
    center_x, center_y = 520, 380

    degrees = np.diff(row_offsets)
    num_levels = len(demo_df)
    ring_radii = np.linspace(60, 320, num_levels + 1)

    node_levels = np.zeros(n, dtype=int)
    nodes_per_level = [int(r["frontier_size"]) for _, r in demo_df.iterrows()]

    assigned = 0
    for lvl, count in enumerate(nodes_per_level):
        end = min(n, assigned + count)
        node_levels[assigned:end] = lvl
        assigned = end
    if assigned < n:
        node_levels[assigned:] = num_levels - 1

    level_counts = {}
    for lvl in range(num_levels):
        level_counts[lvl] = np.sum(node_levels == lvl)

    level_idx = {lvl: 0 for lvl in range(num_levels)}

    for i in range(n):
        lvl = node_levels[i]
        radius = ring_radii[lvl] if lvl < len(ring_radii) else 330
        total_in_lvl = max(1, level_counts.get(lvl, 1))
        idx_in_lvl = level_idx[lvl]
        level_idx[lvl] += 1

        angle = (2.0 * np.pi * idx_in_lvl / total_in_lvl) + (lvl * 0.4)
        r_jitter = radius + (np.random.rand() - 0.5) * 16.0
        x = center_x + r_jitter * np.cos(angle)
        y = center_y + r_jitter * np.sin(angle)
        coords.append({"id": int(i), "x": round(float(x), 1), "y": round(float(y), 1), "level": int(lvl), "deg": int(degrees[i])})

    edges = []
    edge_count = 0
    max_visual_edges = 1200
    for u in range(n):
        start = row_offsets[u]
        end = row_offsets[u + 1]
        for e in range(start, end):
            v = col_indices[e]
            if u < v:
                edges.append({"u": int(u), "v": int(v)})
                edge_count += 1
                if edge_count >= max_visual_edges:
                    break
        if edge_count >= max_visual_edges:
            break

    return coords, edges


def extract_level_records(df_subset):
    """Converts a pandas DataFrame subset into level records for JSON serialization."""
    records = []
    for _, r in df_subset.iterrows():
        records.append({
            "level": int(r["level"]),
            "frontier_size": int(r["frontier_size"]),
            "edge_work": int(r["edge_work"]),
            "mu": round(float(r["mu"]), 2),
            "cv": round(float(r["cv"]), 2),
            "dmax": int(r["dmax"]),
            "mode_chosen": str(r["mode_chosen"]),
            "chunk_size": int(r["chunk_size"]),
            "num_chunks": int(r["num_chunks"]),
            "steals_attempted": int(r["steals_attempted"]),
            "steals_succeeded": int(r["steals_succeeded"]),
            "gpu_batches": int(r["gpu_batches"]),
            "time_us": round(float(r["time_us"]), 1),
            "t_cpu_us": round(float(r["t_cpu_us"]), 1),
            "t_gpu_us": round(float(r["t_gpu_us"]), 1),
            "gpu_idle_us": round(float(r["gpu_idle_us"]), 1)
        })
    return records


def build_fabfs_demo_html(datasets_dict, default_dataset="rmat10_seed1", thresholds_path="config/thresholds.json"):
    """
    Constructs the self-contained, offline multi-graph HTML demo page.
    """
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresh = json.load(f)

    datasets_json = json.dumps(datasets_dict)
    thresh_json = json.dumps(thresh)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FABFS Multi-Graph Adaptive BFS Telemetry & Evaluator Visualizer</title>
<style>
  :root {{
    --bg-dark: #0b0f19;
    --panel-bg: #131b2e;
    --card-bg: #090d16;
    --border-color: #24324d;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent-blue: #38bdf8;
    --accent-green: #4ade80;
    --accent-orange: #fb923c;
    --accent-purple: #c084fc;
    --mode-seq: #38bdf8;
    --mode-static: #4ade80;
    --mode-dynamic: #fb923c;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-dark);
    color: var(--text-primary);
    overflow: hidden;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }}
  header {{
    background: var(--panel-bg);
    border-bottom: 1px solid var(--border-color);
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    z-index: 10;
  }}
  .brand {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .logo {{
    font-weight: 800;
    font-size: 1.25rem;
    letter-spacing: -0.5px;
    color: var(--text-primary);
  }}
  .logo span {{ color: var(--accent-blue); }}
  .tag {{
    font-size: 0.72rem;
    background: #0284c7;
    color: white;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }}
  .graph-select-container {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .graph-select-label {{
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-weight: 600;
    text-transform: uppercase;
  }}
  #graphSelect {{
    background: #090d16;
    color: var(--accent-blue);
    border: 1px solid var(--border-color);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    outline: none;
  }}
  #graphSelect:hover, #graphSelect:focus {{
    border-color: var(--accent-blue);
  }}
  .meta-pills {{
    display: flex;
    gap: 10px;
    font-size: 0.78rem;
    color: var(--text-secondary);
  }}
  .pill {{
    background: #090d16;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
    white-space: nowrap;
  }}
  .pill strong {{ color: var(--text-primary); }}

  .main-container {{
    display: flex;
    flex: 1;
    overflow: hidden;
  }}
  .canvas-panel {{
    flex: 1;
    position: relative;
    background: #060911;
    display: flex;
    flex-direction: column;
  }}
  canvas {{
    display: block;
    width: 100%;
    height: 100%;
  }}
  .side-panel {{
    width: 440px;
    background: var(--panel-bg);
    border-left: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }}
  .panel-section {{
    padding: 14px 18px;
    border-bottom: 1px solid var(--border-color);
  }}
  .panel-title {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .controls-bar {{
    position: absolute;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(19, 27, 46, 0.95);
    border: 1px solid var(--border-color);
    padding: 8px 18px;
    border-radius: 32px;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    backdrop-filter: blur(10px);
    z-index: 5;
  }}
  button {{
    background: #24324d;
    border: none;
    color: white;
    padding: 7px 14px;
    border-radius: 20px;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.82rem;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: background 0.15s;
  }}
  button:hover {{ background: #334568; }}
  button.primary {{ background: #0284c7; }}
  button.primary:hover {{ background: #0369a1; }}
  .level-badge {{
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--accent-blue);
    min-width: 110px;
    text-align: center;
  }}
  .slider-container {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  input[type=range] {{
    -webkit-appearance: none;
    appearance: none;
    width: 130px;
    height: 6px;
    border-radius: 3px;
    background: #24324d;
    outline: none;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--accent-blue);
    cursor: pointer;
    box-shadow: 0 0 6px rgba(56,189,248,0.8);
  }}

  .mode-indicator {{
    display: inline-block;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 1.05rem;
    text-align: center;
    letter-spacing: 0.5px;
    transition: all 0.2s;
  }}
  .mode-SEQ {{ background: rgba(56, 189, 248, 0.15); color: var(--mode-seq); border: 1px solid var(--mode-seq); box-shadow: 0 0 12px rgba(56,189,248,0.25); }}
  .mode-STATIC {{ background: rgba(74, 222, 128, 0.15); color: var(--mode-static); border: 1px solid var(--mode-static); box-shadow: 0 0 12px rgba(74,222,128,0.25); }}
  .mode-DYNAMIC {{ background: rgba(251, 146, 60, 0.15); color: var(--mode-dynamic); border: 1px solid var(--mode-dynamic); box-shadow: 0 0 12px rgba(251,146,60,0.25); }}

  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }}
  .stat-card {{
    background: var(--card-bg);
    padding: 9px 12px;
    border-radius: 8px;
    border: 1px solid var(--border-color);
  }}
  .stat-label {{ font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 3px; }}
  .stat-value {{ font-size: 1.05rem; font-weight: 700; color: var(--text-primary); }}
  .stat-sub {{ font-size: 0.68rem; color: var(--text-secondary); margin-top: 2px; }}

  .decision-box {{
    background: var(--card-bg);
    border-left: 4px solid var(--accent-blue);
    padding: 11px 13px;
    border-radius: 4px;
    font-size: 0.8rem;
    line-height: 1.45;
    color: var(--text-secondary);
  }}
  .decision-box strong {{ color: var(--text-primary); }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.74rem;
    margin-bottom: 5px;
  }}
  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }}
  .toast {{
    position: absolute;
    top: 16px;
    left: 20px;
    background: rgba(9, 13, 22, 0.9);
    border: 1px solid var(--border-color);
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    backdrop-filter: blur(6px);
    pointer-events: none;
    z-index: 5;
  }}
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="logo">FABFS <span>Visualizer</span></div>
    <div class="tag">M10 Multi-Graph</div>
  </div>

  <div class="graph-select-container">
    <div class="graph-select-label">Select Graph:</div>
    <select id="graphSelect" onchange="switchDataset(this.value)">
      <option value="rmat10_seed1">rmat10_seed1 (1K Nodes — 2D Topology)</option>
      <option value="web-Google">web-Google (875K Nodes — Phase Transitions)</option>
      <option value="com-lj">com-lj (4.0M Nodes — Heavy Hub Skew)</option>
      <option value="roadNet-CA">roadNet-CA (1.9M Nodes — High Diameter Planar)</option>
    </select>
  </div>

  <div class="meta-pills">
    <div class="pill">Graph: <strong id="pillGraph">rmat10_seed1</strong></div>
    <div class="pill">Run ID: <strong id="pillRunId">demo_rmat10_run1</strong></div>
    <div class="pill">Vertices: <strong id="pillVertices">1,024</strong></div>
    <div class="pill">Edges: <strong id="pillEdges">4,960</strong></div>
    <div class="pill">Policy: <strong>Adaptive</strong></div>
  </div>
</header>

<div class="main-container">
  <div class="canvas-panel">
    <div class="toast" id="viewModeToast">
      2D Concentric Topology View &bull; Step through levels to watch frontier pulse and mode transitions.
    </div>
    <canvas id="graphCanvas"></canvas>
    
    <div class="controls-bar">
      <button onclick="prevLevel()">&#9664; Prev</button>
      <button class="primary" id="playBtn" onclick="togglePlay()">&#9654; Play</button>
      <button onclick="nextLevel()">Next &#9654;</button>
      <button onclick="resetBFS()">&#8634; Reset</button>
      <div class="slider-container">
        <input type="range" id="levelSlider" min="0" max="3" value="0" oninput="seekLevel(parseInt(this.value))">
      </div>
      <div class="level-badge" id="levelIndicator">Level 0 / 3</div>
    </div>
  </div>

  <div class="side-panel">
    <div class="panel-section">
      <div class="panel-title">Adaptive Decision Rule (Algorithm 1)</div>
      <div style="text-align: center; margin-bottom: 10px;">
        <div id="modeChosen" class="mode-indicator mode-SEQ">SEQ MODE</div>
      </div>
      <div class="decision-box" id="decisionExplanation">
        Frontier size |F| &lt; T_min (1024). Single CPU thread expands frontier without parallelization overhead.
      </div>
    </div>

    <div class="panel-section">
      <div class="panel-title">Frontier Topology Telemetry</div>
      <div class="grid-2">
        <div class="stat-card">
          <div class="stat-label">Frontier Size |F_l|</div>
          <div class="stat-value" id="valFrontier">0</div>
          <div class="stat-sub">Active vertices in level</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Active Edges |E_l|</div>
          <div class="stat-value" id="valEdges">0</div>
          <div class="stat-sub">Degree sum in F_l</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Degree Skew (CV)</div>
          <div class="stat-value" id="valCV">0.00</div>
          <div class="stat-sub">Threshold T_CV = {thresh.get('T_CV', 1.5)}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Max Degree (dmax)</div>
          <div class="stat-value" id="valDmax">0</div>
          <div class="stat-sub">Peak hub degree in F_l</div>
        </div>
      </div>
    </div>

    <div class="panel-section">
      <div class="panel-title">Dispatch & Scheduling Metrics</div>
      <div class="grid-2">
        <div class="stat-card">
          <div class="stat-label">Chunk Size (C_l)</div>
          <div class="stat-value" id="valChunkSize">0</div>
          <div class="stat-sub" id="valNumChunks">0 chunks</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Work Stealing</div>
          <div class="stat-value" id="valSteals">0 / 0</div>
          <div class="stat-sub">Succeeded / Attempted</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Level Time</div>
          <div class="stat-value" id="valTime">0.0 us</div>
          <div class="stat-sub">CPU: <span id="valCpuTime">0</span> us | GPU: <span id="valGpuTime">0</span> us</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">GPU Idle Time</div>
          <div class="stat-value" id="valGpuIdle">0.0 us</div>
          <div class="stat-sub">Proxy stall overhead</div>
        </div>
      </div>
    </div>

    <div class="panel-section">
      <div class="panel-title">Calibration Constants (Frozen Gate 1)</div>
      <div style="font-size: 0.74rem; color: var(--text-secondary); line-height: 1.6;">
        <div>&bull; T_min = <strong>{thresh.get('T_min', 1024)}</strong> vertices (Sequential threshold)</div>
        <div>&bull; T_CV = <strong>{thresh.get('T_CV', 1.5)}</strong> (Degree skew threshold)</div>
        <div>&bull; T_par = <strong>{thresh.get('T_par', 8192)}</strong> vertices (Parallel cutoff)</div>
        <div>&bull; Static GPU Share (&alpha;) = <strong>{thresh.get('alpha', 0.1526)}</strong></div>
        <div>&bull; Deque Steal Latency (C_steal) = <strong>{thresh.get('C_steal_ns', 135.6)} ns</strong></div>
        <div>&bull; GPU Kernel Launch Latency (L) = <strong>{thresh.get('L_launch_us', 39.99)} us</strong></div>
      </div>
    </div>

    <div class="panel-section">
      <div class="panel-title">Visual Legend</div>
      <div class="legend-item"><div class="legend-dot" style="background: #38bdf8;"></div> <strong>SEQ Mode:</strong> Single CPU Thread (|F| &lt; 1,024)</div>
      <div class="legend-item"><div class="legend-dot" style="background: #4ade80;"></div> <strong>STATIC Mode:</strong> Edge-Balanced CPU/GPU Split</div>
      <div class="legend-item"><div class="legend-dot" style="background: #fb923c;"></div> <strong>DYNAMIC Mode:</strong> Chase-Lev Work-Stealing</div>
      <div class="legend-item"><div class="legend-dot" style="background: #f59e0b;"></div> Source Vertex (Level 0)</div>
      <div class="legend-item"><div class="legend-dot" style="background: #334155;"></div> Unvisited Vertices / Inactive Levels</div>
    </div>
  </div>
</div>

<script>
  const allDatasets = {datasets_json};
  const thresholds = {thresh_json};

  let activeDatasetKey = "{default_dataset}";
  let currentLevel = 0;
  let isPlaying = false;
  let timer = null;

  const canvas = document.getElementById("graphCanvas");
  const ctx = canvas.getContext("2d");

  function getActiveDataset() {{
    return allDatasets[activeDatasetKey];
  }}

  function switchDataset(key) {{
    if (isPlaying) togglePlay();
    activeDatasetKey = key;
    currentLevel = 0;

    const ds = getActiveDataset();
    document.getElementById("pillGraph").innerText = ds.name;
    document.getElementById("pillRunId").innerText = ds.run_id;
    document.getElementById("pillVertices").innerText = Number(ds.nodes).toLocaleString();
    document.getElementById("pillEdges").innerText = Number(ds.edges).toLocaleString();

    const slider = document.getElementById("levelSlider");
    slider.max = ds.levels.length - 1;
    slider.value = 0;

    const toast = document.getElementById("viewModeToast");
    if (ds.has_node_layout) {{
      toast.innerHTML = `<strong>2D Concentric Topology View:</strong> ${{ds.name}} &bull; ${{ds.nodes}} nodes &bull; Step or Play to watch frontier expansion.`;
    }} else {{
      toast.innerHTML = `<strong>Frontier Wave & Mode Profile:</strong> ${{ds.name}} (${{ds.category}}) &bull; ${{ds.nodes.toLocaleString()}} vertices &bull; Click any level bar to inspect telemetry.`;
    }}

    updateHUD();
    drawScene();
  }}

  function resizeCanvas() {{
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    drawScene();
  }}
  window.addEventListener("resize", resizeCanvas);

  // Click on canvas to jump directly to level when in Wave Profile view
  canvas.addEventListener("click", (evt) => {{
    const ds = getActiveDataset();
    if (ds.has_node_layout) return;

    const rect = canvas.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;

    const chartLeft = 50;
    const chartRight = canvas.width - 50;
    const chartTop = 60;
    const chartBottom = canvas.height * 0.62;

    if (x >= chartLeft && x <= chartRight && y >= chartTop && y <= chartBottom) {{
      const numLevels = ds.levels.length;
      const barWidth = (chartRight - chartLeft) / numLevels;
      const clickedIdx = Math.floor((x - chartLeft) / barWidth);
      if (clickedIdx >= 0 && clickedIdx < numLevels) {{
        seekLevel(clickedIdx);
      }}
    }}
  }});

  function drawScene() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const ds = getActiveDataset();

    if (ds.has_node_layout) {{
      drawGraphTopology(ds);
    }} else {{
      drawFrontierWaveProfile(ds);
    }}
  }}

  function drawGraphTopology(ds) {{
    const coordsData = ds.coords;
    const edgesData = ds.edges_sample;
    const levelsData = ds.levels;

    const scaleX = canvas.width / 1050;
    const scaleY = canvas.height / 760;

    // Draw background edges
    ctx.strokeStyle = "rgba(51, 65, 85, 0.35)";
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    for (let i = 0; i < edgesData.length; ++i) {{
      const e = edgesData[i];
      const u = coordsData[e.u];
      const v = coordsData[e.v];
      if (u && v) {{
        ctx.moveTo(u.x * scaleX, u.y * scaleY);
        ctx.lineTo(v.x * scaleX, v.y * scaleY);
      }}
    }}
    ctx.stroke();

    // Draw nodes
    for (let i = 0; i < coordsData.length; ++i) {{
      const n = coordsData[i];
      const px = n.x * scaleX;
      const py = n.y * scaleY;

      ctx.beginPath();
      if (n.id === 0) {{
        ctx.arc(px, py, 6, 0, 2 * Math.PI);
        ctx.fillStyle = "#f59e0b";
        ctx.shadowColor = "#f59e0b";
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
      }} else if (n.level < currentLevel) {{
        ctx.arc(px, py, 2.8, 0, 2 * Math.PI);
        const lMode = levelsData[n.level] ? levelsData[n.level].mode_chosen : "SEQ";
        ctx.fillStyle = lMode === "STATIC" ? "#4ade80" : (lMode === "DYNAMIC" ? "#fb923c" : "#38bdf8");
        ctx.fill();
      }} else if (n.level === currentLevel) {{
        ctx.arc(px, py, 4.8, 0, 2 * Math.PI);
        const lMode = levelsData[currentLevel] ? levelsData[currentLevel].mode_chosen : "SEQ";
        ctx.fillStyle = lMode === "STATIC" ? "#4ade80" : (lMode === "DYNAMIC" ? "#fb923c" : "#38bdf8");
        ctx.shadowColor = ctx.fillStyle;
        ctx.shadowBlur = 9;
        ctx.fill();
        ctx.shadowBlur = 0;
      }} else {{
        ctx.arc(px, py, 1.8, 0, 2 * Math.PI);
        ctx.fillStyle = "#24324d";
        ctx.fill();
      }}
    }}
  }}

  function drawFrontierWaveProfile(ds) {{
    const levels = ds.levels;
    const numLevels = levels.length;

    const chartLeft = 50;
    const chartRight = canvas.width - 50;
    const chartTop = 70;
    const chartBottom = canvas.height * 0.60;
    const chartWidth = chartRight - chartLeft;
    const chartHeight = chartBottom - chartTop;

    // Header Title on Canvas
    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 15px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.fillText(`${{ds.name}} &mdash; Dynamic Frontier Wave & Mode Allocation`, chartLeft, 38);

    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.fillText(`${{numLevels}} BFS Levels &bull; Click on any bar to jump &bull; Bar height: log10(|F_l| + 1)`, chartLeft, 55);

    // Baseline grid line
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(chartLeft, chartBottom);
    ctx.lineTo(chartRight, chartBottom);
    ctx.stroke();

    // Max log value for scaling
    let maxLogF = 1;
    for (let l of levels) {{
      const val = Math.log10(l.frontier_size + 1);
      if (val > maxLogF) maxLogF = val;
    }}
    if (maxLogF < 1) maxLogF = 1;

    const barWidth = chartWidth / numLevels;
    const barPad = numLevels > 50 ? 0 : Math.max(1, barWidth * 0.15);

    // Draw bars
    for (let i = 0; i < numLevels; ++i) {{
      const lvl = levels[i];
      const logF = Math.log10(lvl.frontier_size + 1);
      const h = Math.max(4, (logF / maxLogF) * (chartHeight - 30));
      const x = chartLeft + i * barWidth + barPad;
      const w = Math.max(1.5, barWidth - 2 * barPad);
      const y = chartBottom - h;

      let color = "#38bdf8"; // SEQ
      if (lvl.mode_chosen === "STATIC") color = "#4ade80";
      else if (lvl.mode_chosen === "DYNAMIC") color = "#fb923c";

      ctx.fillStyle = color;
      if (i === currentLevel) {{
        // Highlight active level bar
        ctx.fillStyle = "#ffffff";
        ctx.shadowColor = color;
        ctx.shadowBlur = 14;
        ctx.fillRect(x - 1, y - 4, w + 2, h + 4);
        ctx.shadowBlur = 0;

        // Pointer triangle above bar
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(x + w / 2, y - 8);
        ctx.lineTo(x + w / 2 - 5, y - 16);
        ctx.lineTo(x + w / 2 + 5, y - 16);
        ctx.closePath();
        ctx.fill();
      }} else if (i < currentLevel) {{
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 1.0;
      }} else {{
        ctx.globalAlpha = 0.35;
        ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 1.0;
      }}
    }}

    // Degree Skew (CV) Curve overlay
    ctx.beginPath();
    ctx.strokeStyle = "rgba(192, 132, 252, 0.75)"; // accent purple
    ctx.lineWidth = 1.8;
    const maxCV = 7.0;
    for (let i = 0; i < numLevels; ++i) {{
      const lvl = levels[i];
      const cx = chartLeft + i * barWidth + barWidth / 2;
      const cy = chartBottom - (Math.min(lvl.cv, maxCV) / maxCV) * (chartHeight - 40);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }}
    ctx.stroke();

    // Threshold line T_CV = 1.5
    const cvThreshY = chartBottom - (1.5 / maxCV) * (chartHeight - 40);
    ctx.strokeStyle = "rgba(251, 146, 60, 0.5)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(chartLeft, cvThreshY);
    ctx.lineTo(chartRight, cvThreshY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(251, 146, 60, 0.8)";
    ctx.font = "11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.fillText("T_CV = 1.5", chartRight - 55, cvThreshY - 4);

    // Active Engine Architecture Schematic Diagram in Lower Canvas
    drawEngineArchitectureDiagram(chartLeft, chartRight, chartBottom + 25, canvas.height - 75, levels[currentLevel]);
  }}

  function drawEngineArchitectureDiagram(left, right, top, bottom, lvlData) {{
    const w = right - left;
    const h = bottom - top;

    // Background card for schematic
    ctx.fillStyle = "#0c1220";
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(left, top, w, h, 8);
    ctx.fill();
    ctx.stroke();

    // Schematic title
    ctx.fillStyle = "#94a3b8";
    ctx.font = "bold 12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.fillText("ACTIVE EXECUTION DISPATCH PIPELINE", left + 16, top + 22);

    const mode = lvlData ? lvlData.mode_chosen : "SEQ";

    if (mode === "SEQ") {{
      // Diagram: Single CPU Thread pipeline
      ctx.fillStyle = "#38bdf8";
      ctx.font = "bold 13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText("Single-Threaded Sequential CPU Expansion (Rule 1: |F| < 1,024)", left + 16, top + 44);

      ctx.fillStyle = "#94a3b8";
      ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText("Frontier fits in L1/L2 cache &bull; Deque lock & GPU launch overhead strictly eliminated.", left + 16, top + 64);

      // Core box
      drawBox(left + 20, top + 76, 160, 34, "#38bdf8", "Thread 0 (Core 0)");
      drawArrow(left + 185, top + 93, left + 230, top + 93, "#38bdf8");
      drawBox(left + 235, top + 76, 200, 34, "#38bdf8", "Direct Bitvector Visitor");
      drawArrow(left + 440, top + 93, left + 485, top + 93, "#38bdf8");
      drawBox(left + 490, top + 76, 180, 34, "#38bdf8", "Zero Sync Overhead");
    }} else if (mode === "STATIC") {{
      // Diagram: Edge-Balanced Static CPU/GPU Split
      ctx.fillStyle = "#4ade80";
      ctx.font = "bold 13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText("Deterministic Dual-Device Edge Split (Rule 2: Low Skew CV <= 1.5 or |F| <= 8,192)", left + 16, top + 44);

      ctx.fillStyle = "#94a3b8";
      ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText("Edge-balanced partition: GPU handles alpha=15.3% bulk edges &bull; CPU threads process remainder.", left + 16, top + 64);

      drawBox(left + 20, top + 76, 180, 34, "#4ade80", "CPU Pool (84.7% Edges)");
      drawBox(left + 215, top + 76, 200, 34, "#4ade80", "GPU RTX 4050 (15.3% Edges)");
      drawArrow(left + 420, top + 93, left + 465, top + 93, "#4ade80");
      drawBox(left + 470, top + 76, 200, 34, "#4ade80", "Global Barrier & Flip");
    }} else {{
      // Diagram: Dynamic Work Stealing
      ctx.fillStyle = "#fb923c";
      ctx.font = "bold 13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText("Chase-Lev Work-Stealing Pool with Dynamic Chunking (Rule 3: High Skew CV > 1.5)", left + 16, top + 44);

      ctx.fillStyle = "#94a3b8";
      ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText(`Dynamic chunks C_l=${{lvlData.chunk_size}} &bull; Steal attempts: ${{lvlData.steals_attempted}} &bull; Succeeded: ${{lvlData.steals_succeeded}} &bull; Eliminates hub contention.`, left + 16, top + 64);

      drawBox(left + 20, top + 76, 170, 34, "#fb923c", `Lock-Free Deques (C_l=${{lvlData.chunk_size}})`);
      drawArrow(left + 195, top + 93, left + 235, top + 93, "#fb923c");
      drawBox(left + 240, top + 76, 210, 34, "#fb923c", `8 Stealing Workers`);
      drawArrow(left + 455, top + 93, left + 495, top + 93, "#fb923c");
      drawBox(left + 500, top + 76, 180, 34, "#fb923c", "Coalesced Frontier GPU Proxy");
    }}
  }}

  function drawBox(x, y, w, h, color, text) {{
    ctx.strokeStyle = color;
    ctx.fillStyle = "rgba(19, 27, 46, 0.9)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 6);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(text, x + w / 2, y + h / 2 + 4);
    ctx.textAlign = "start";
  }}

  function drawArrow(x1, y1, x2, y2, color) {{
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 5, y2 - 4);
    ctx.lineTo(x2 - 5, y2 + 4);
    ctx.closePath();
    ctx.fill();
  }}

  function updateHUD() {{
    const ds = getActiveDataset();
    const data = ds.levels[currentLevel];
    if (!data) return;

    document.getElementById("levelIndicator").innerText = `Level ${{currentLevel}} / ${{ds.levels.length - 1}}`;
    document.getElementById("levelSlider").value = currentLevel;

    const modeEl = document.getElementById("modeChosen");
    modeEl.className = `mode-indicator mode-${{data.mode_chosen}}`;
    modeEl.innerText = `${{data.mode_chosen}} MODE`;

    document.getElementById("valFrontier").innerText = Number(data.frontier_size).toLocaleString();
    document.getElementById("valEdges").innerText = Number(data.edge_work).toLocaleString();
    document.getElementById("valCV").innerText = Number(data.cv).toFixed(2);
    document.getElementById("valDmax").innerText = Number(data.dmax).toLocaleString();

    document.getElementById("valChunkSize").innerText = data.chunk_size ? data.chunk_size : "N/A";
    document.getElementById("valNumChunks").innerText = data.num_chunks ? `${{data.num_chunks}} chunks` : "0 chunks";
    document.getElementById("valSteals").innerText = `${{data.steals_succeeded}} / ${{data.steals_attempted}}`;

    document.getElementById("valTime").innerText = `${{data.time_us}} us`;
    document.getElementById("valCpuTime").innerText = data.t_cpu_us;
    document.getElementById("valGpuTime").innerText = data.t_gpu_us;
    document.getElementById("valGpuIdle").innerText = `${{data.gpu_idle_us}} us`;

    // Algorithm 1 Rule Explanation
    const expl = document.getElementById("decisionExplanation");
    if (data.mode_chosen === "SEQ") {{
      expl.innerHTML = `<strong>Rule 1 Applied:</strong> Frontier size |F| (${{Number(data.frontier_size).toLocaleString()}}) &lt; T_min (${{thresholds.T_min}}). Executed by single CPU thread to eliminate parallel synchronization overhead.`;
    }} else if (data.mode_chosen === "STATIC") {{
      expl.innerHTML = `<strong>Rule 2 Applied:</strong> Frontier size |F| (${{Number(data.frontier_size).toLocaleString()}}) &le; T_par (${{thresholds.T_par}}) or CV (${{data.cv}}) &le; T_CV (${{thresholds.T_CV}}). Deterministic edge-balanced split: GPU receives &alpha;=${{thresholds.alpha}} work.`;
    }} else {{
      expl.innerHTML = `<strong>Rule 3 Applied:</strong> High degree skew CV (${{data.cv}}) &gt; T_CV (${{thresholds.T_CV}}) and |F| &gt; T_par (${{thresholds.T_par}}). Dispatched to Chase-Lev work-stealing with dynamic chunking (C_l=${{data.chunk_size}}).`;
    }}
  }}

  function seekLevel(lvl) {{
    const ds = getActiveDataset();
    if (lvl >= 0 && lvl < ds.levels.length) {{
      currentLevel = lvl;
      updateHUD();
      drawScene();
    }}
  }}

  function nextLevel() {{
    const ds = getActiveDataset();
    if (currentLevel < ds.levels.length - 1) {{
      currentLevel++;
      updateHUD();
      drawScene();
    }} else if (isPlaying) {{
      togglePlay();
    }}
  }}

  function prevLevel() {{
    if (currentLevel > 0) {{
      currentLevel--;
      updateHUD();
      drawScene();
    }}
  }}

  function resetBFS() {{
    if (isPlaying) togglePlay();
    currentLevel = 0;
    updateHUD();
    drawScene();
  }}

  function togglePlay() {{
    isPlaying = !isPlaying;
    const btn = document.getElementById("playBtn");
    if (isPlaying) {{
      btn.innerHTML = "&#10074;&#10074; Pause";
      btn.style.background = "#e11d48";
      const ds = getActiveDataset();
      const intervalMs = ds.levels.length > 50 ? 80 : 800;
      timer = setInterval(nextLevel, intervalMs);
    }} else {{
      btn.innerHTML = "&#9654; Play";
      btn.style.background = "#0284c7";
      clearInterval(timer);
    }}
  }}

  // Initialize
  window.onload = () => {{
    switchDataset("{default_dataset}");
    resizeCanvas();
  }};
</script>
</body>
</html>
"""
    out_path = "demo/fabfs_demo.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[build_demo.py] Successfully wrote self-contained multi-graph HTML demo: {out_path} ({len(html_content)} bytes)")
    return out_path


def main():
    print("[build_demo.py] Validating demo run and generating multi-graph visualizer...")
    demo_df, rmat_run_id, rmat_graph_name = validate_demo_run()
    bin_path = "demo/demo_graph.bin"
    n, m, row_offsets, col_indices = load_binary_graph(bin_path)
    print(f"[build_demo.py] Loaded {bin_path}: n={n}, m={m}")

    coords, edges_sample = generate_node_layout(n, row_offsets, col_indices, demo_df)

    # Read full levels.csv to extract real evaluation graph traces
    levels_df = pd.read_csv("results/levels.csv")

    datasets_dict = {}

    # 1. rmat10_seed1 (Synthetic R-MAT with 2D node layout)
    datasets_dict["rmat10_seed1"] = {
        "key": "rmat10_seed1",
        "name": "rmat10_seed1",
        "category": "Synthetic R-MAT (Scale 10)",
        "nodes": int(n),
        "edges": int(m),
        "run_id": rmat_run_id,
        "has_node_layout": True,
        "coords": coords,
        "edges_sample": edges_sample,
        "levels": extract_level_records(demo_df)
    }

    # 2. web-Google (Real web crawl with SEQ -> STATIC -> DYNAMIC -> STATIC -> SEQ)
    web_run_id = "run_062"
    web_sub = levels_df[levels_df["run_id"] == web_run_id]
    if len(web_sub) > 0:
        datasets_dict["web-Google"] = {
            "key": "web-Google",
            "name": "web-Google",
            "category": "Web Crawl Graph",
            "nodes": 875713,
            "edges": 5105039,
            "run_id": web_run_id,
            "has_node_layout": False,
            "levels": extract_level_records(web_sub)
        }
        print(f"[build_demo.py] Embedded web-Google ({web_run_id}): {len(web_sub)} levels")

    # 3. com-lj (LiveJournal social network with heavy degree skew)
    lj_run_id = "run_126"
    lj_sub = levels_df[levels_df["run_id"] == lj_run_id]
    if len(lj_sub) > 0:
        datasets_dict["com-lj"] = {
            "key": "com-lj",
            "name": "com-lj",
            "category": "Social Network (LiveJournal)",
            "nodes": 3997962,
            "edges": 34681189,
            "run_id": lj_run_id,
            "has_node_layout": False,
            "levels": extract_level_records(lj_sub)
        }
        print(f"[build_demo.py] Embedded com-lj ({lj_run_id}): {len(lj_sub)} levels")

    # 4. roadNet-CA (California road network, planar, 555 levels)
    road_run_id = "run_005"
    road_sub = levels_df[levels_df["run_id"] == road_run_id]
    if len(road_sub) > 0:
        datasets_dict["roadNet-CA"] = {
            "key": "roadNet-CA",
            "name": "roadNet-CA",
            "category": "Planar Road Network",
            "nodes": 1965206,
            "edges": 5533214,
            "run_id": road_run_id,
            "has_node_layout": False,
            "levels": extract_level_records(road_sub)
        }
        print(f"[build_demo.py] Embedded roadNet-CA ({road_run_id}): {len(road_sub)} levels")

    out_html = build_fabfs_demo_html(datasets_dict, default_dataset="rmat10_seed1")
    print(f"[build_demo.py] Multi-graph demo build complete: {out_html}")


if __name__ == "__main__":
    main()
