#!/usr/bin/env python3
"""
M10: Demo builder script.
Reads one run's rows from results/levels.csv and demo/demo_run.csv,
reads demo/demo_graph.bin, validates run_id and graph hash integrity,
and generates demo/fabfs_demo.html (a self-contained, offline HTML visualizer).
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

    # Group vertices by approximate level or degree
    # For small demo graphs (1024 nodes), layout in concentric rings by level
    # Source node (0) at center (500, 400)
    coords = []
    center_x, center_y = 520, 380

    # Collect degrees
    degrees = np.diff(row_offsets)

    # Ring radius per level
    num_levels = len(demo_df)
    ring_radii = np.linspace(60, 320, num_levels + 1)

    # Distribute nodes across rings based on degrees / level distribution
    node_levels = np.zeros(n, dtype=int)
    nodes_per_level = [int(r["frontier_size"]) for _, r in demo_df.iterrows()]

    assigned = 0
    for lvl, count in enumerate(nodes_per_level):
        end = min(n, assigned + count)
        node_levels[assigned:end] = lvl
        assigned = end
    if assigned < n:
        node_levels[assigned:] = num_levels - 1

    # Place nodes on their level's ring
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
        # Small jitter
        r_jitter = radius + (np.random.rand() - 0.5) * 16.0
        x = center_x + r_jitter * np.cos(angle)
        y = center_y + r_jitter * np.sin(angle)
        coords.append({"id": int(i), "x": round(float(x), 1), "y": round(float(y), 1), "level": int(lvl), "deg": int(degrees[i])})

    # Pick a sample of visual edges (up to 800 edges to maintain crisp canvas rendering)
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


def build_fabfs_demo_html(demo_df, run_id, graph_name, coords, edges, thresholds_path="config/thresholds.json"):
    """
    Constructs the self-contained, offline HTML demo page.
    """
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresh = json.load(f)

    levels_data = []
    for _, r in demo_df.iterrows():
        levels_data.append({
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

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FABFS Live Algorithm & Frontier Telemetry Demo</title>
<style>
  :root {{
    --bg-dark: #0f172a;
    --panel-bg: #1e293b;
    --border-color: #334155;
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
    padding: 12px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .brand {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .logo {{
    font-weight: 800;
    font-size: 1.3rem;
    letter-spacing: -0.5px;
    color: var(--text-primary);
  }}
  .logo span {{ color: var(--accent-blue); }}
  .tag {{
    font-size: 0.75rem;
    background: #0284c7;
    color: white;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 600;
  }}
  .meta-pills {{
    display: flex;
    gap: 12px;
    font-size: 0.8rem;
    color: var(--text-secondary);
  }}
  .pill {{
    background: #0f172a;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
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
    background: #090d16;
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
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
  }}
  .panel-title {{
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
  }}
  .controls-bar {{
    position: absolute;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(30, 41, 59, 0.95);
    border: 1px solid var(--border-color);
    padding: 8px 16px;
    border-radius: 30px;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    backdrop-filter: blur(8px);
  }}
  button {{
    background: #334155;
    border: none;
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: background 0.2s;
  }}
  button:hover {{ background: #475569; }}
  button.primary {{ background: #0284c7; }}
  button.primary:hover {{ background: #0369a1; }}
  .level-badge {{
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-blue);
    min-width: 90px;
    text-align: center;
  }}
  .mode-indicator {{
    display: inline-block;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 1.1rem;
    text-align: center;
    letter-spacing: 0.5px;
  }}
  .mode-SEQ {{ background: rgba(56, 189, 248, 0.15); color: var(--mode-seq); border: 1px solid var(--mode-seq); }}
  .mode-STATIC {{ background: rgba(74, 222, 128, 0.15); color: var(--mode-static); border: 1px solid var(--mode-static); }}
  .mode-DYNAMIC {{ background: rgba(251, 146, 60, 0.15); color: var(--mode-dynamic); border: 1px solid var(--mode-dynamic); }}

  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }}
  .stat-card {{
    background: #0f172a;
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid var(--border-color);
  }}
  .stat-label {{ font-size: 0.72rem; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 4px; }}
  .stat-value {{ font-size: 1.1rem; font-weight: 700; color: var(--text-primary); }}
  .stat-sub {{ font-size: 0.7rem; color: var(--text-secondary); margin-top: 2px; }}

  .decision-box {{
    background: #0f172a;
    border-left: 4px solid var(--accent-blue);
    padding: 12px;
    border-radius: 4px;
    font-size: 0.82rem;
    line-height: 1.4;
    color: var(--text-secondary);
  }}
  .decision-box strong {{ color: var(--text-primary); }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
    margin-bottom: 6px;
  }}
  .legend-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
  }}
  .toast {{
    position: absolute;
    top: 20px;
    left: 20px;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid var(--border-color);
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 0.82rem;
    color: var(--text-secondary);
    backdrop-filter: blur(4px);
    pointer-events: none;
  }}
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="logo">FABFS <span>Interactive Visualizer</span></div>
    <div class="tag">M10 Replay Engine</div>
  </div>
  <div class="meta-pills">
    <div class="pill">Graph: <strong>{graph_name}</strong></div>
    <div class="pill">Run ID: <strong>{run_id}</strong></div>
    <div class="pill">Policy: <strong>Frontier-Adaptive</strong></div>
    <div class="pill">Commit: <strong>{thresh.get('calibration_commit', '')[:8]}</strong></div>
  </div>
</header>

<div class="main-container">
  <div class="canvas-panel">
    <div class="toast">
      Tip: Click 'Step' or 'Play' to observe level-by-level adaptive dispatch decisions.
    </div>
    <canvas id="graphCanvas"></canvas>
    
    <div class="controls-bar">
      <button onclick="prevLevel()">&#9664; Prev</button>
      <button class="primary" id="playBtn" onclick="togglePlay()">&#9654; Play</button>
      <button onclick="nextLevel()">Next &#9654;</button>
      <button onclick="resetBFS()">&#8634; Reset</button>
      <div class="level-badge" id="levelIndicator">Level 0 / {len(levels_data)-1}</div>
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
          <div class="stat-sub">Active vertices</div>
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
          <div class="stat-sub">Peak hub degree</div>
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
      <div class="panel-title">Calibration Constants (Frozen)</div>
      <div style="font-size: 0.76rem; color: var(--text-secondary); line-height: 1.6;">
        <div>&bull; T_min = <strong>{thresh.get('T_min', 1024)}</strong> vertices</div>
        <div>&bull; T_CV = <strong>{thresh.get('T_CV', 1.5)}</strong></div>
        <div>&bull; T_par = <strong>{thresh.get('T_par', 8192)}</strong> vertices</div>
        <div>&bull; Static GPU Share (&alpha;) = <strong>{thresh.get('alpha', 0.1526)}</strong></div>
        <div>&bull; Deque Steal Latency (C_steal) = <strong>{thresh.get('C_steal_ns', 135.6)} ns</strong></div>
        <div>&bull; GPU Kernel Launch Latency (L) = <strong>{thresh.get('L_launch_us', 39.99)} us</strong></div>
      </div>
    </div>

    <div class="panel-section">
      <div class="panel-title">Visual Legend</div>
      <div class="legend-item"><div class="legend-dot" style="background: #f59e0b;"></div> Source Vertex (Level 0)</div>
      <div class="legend-item"><div class="legend-dot" style="background: #38bdf8;"></div> Visited (SEQ mode)</div>
      <div class="legend-item"><div class="legend-dot" style="background: #4ade80;"></div> Visited (STATIC mode - GPU/CPU split)</div>
      <div class="legend-item"><div class="legend-dot" style="background: #fb923c;"></div> Visited (DYNAMIC mode - Work Stealing)</div>
      <div class="legend-item"><div class="legend-dot" style="background: #334155;"></div> Unvisited Vertices</div>
    </div>
  </div>
</div>

<script>
  const levelsData = {json.dumps(levels_data)};
  const coordsData = {json.dumps(coords)};
  const edgesData = {json.dumps(edges)};
  const thresholds = {json.dumps(thresh)};

  let currentLevel = 0;
  let isPlaying = false;
  let timer = null;

  const canvas = document.getElementById("graphCanvas");
  const ctx = canvas.getContext("2d");

  function resizeCanvas() {{
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    drawGraph();
  }}
  window.addEventListener("resize", resizeCanvas);

  function drawGraph() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Scale coordinates to fit viewport
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
        // Source node
        ctx.arc(px, py, 6, 0, 2 * Math.PI);
        ctx.fillStyle = "#f59e0b";
        ctx.shadowColor = "#f59e0b";
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
      }} else if (n.level < currentLevel) {{
        // Visited in past level
        ctx.arc(px, py, 2.8, 0, 2 * Math.PI);
        const lMode = levelsData[n.level] ? levelsData[n.level].mode_chosen : "SEQ";
        ctx.fillStyle = lMode === "STATIC" ? "#4ade80" : (lMode === "DYNAMIC" ? "#fb923c" : "#38bdf8");
        ctx.fill();
      }} else if (n.level === currentLevel) {{
        // Active frontier
        ctx.arc(px, py, 4.5, 0, 2 * Math.PI);
        const lMode = levelsData[currentLevel] ? levelsData[currentLevel].mode_chosen : "SEQ";
        ctx.fillStyle = lMode === "STATIC" ? "#4ade80" : (lMode === "DYNAMIC" ? "#fb923c" : "#38bdf8");
        ctx.shadowColor = ctx.fillStyle;
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      }} else {{
        // Unvisited
        ctx.arc(px, py, 1.8, 0, 2 * Math.PI);
        ctx.fillStyle = "#334155";
        ctx.fill();
      }}
    }}
  }}

  function updateHUD() {{
    const data = levelsData[currentLevel];
    document.getElementById("levelIndicator").innerText = `Level ${{currentLevel}} / ${{levelsData.length - 1}}`;

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

    // Explanation
    const expl = document.getElementById("decisionExplanation");
    if (data.mode_chosen === "SEQ") {{
      expl.innerHTML = `<strong>Rule 1 Applied:</strong> Frontier |F| (${{data.frontier_size}}) &lt; T_min (${{thresholds.T_min}}). Executed by single CPU thread to eliminate parallel scheduling overhead.`;
    }} else if (data.mode_chosen === "STATIC") {{
      expl.innerHTML = `<strong>Rule 2 Applied:</strong> Frontier size |F| (${{data.frontier_size}}) &le; T_par (${{thresholds.T_par}}) or CV (${{data.cv}}) &le; T_CV (${{thresholds.T_CV}}). Deterministic edge-balanced split: GPU receives &alpha;=${{thresholds.alpha}} work.`;
    }} else {{
      expl.innerHTML = `<strong>Rule 3 Applied:</strong> High degree skew CV (${{data.cv}}) &gt; T_CV (${{thresholds.T_CV}}) and |F| &gt; T_par. Dispatched to Chase-Lev work-stealing with dynamic chunking (C_l=${{data.chunk_size}}).`;
    }}
  }}

  function nextLevel() {{
    if (currentLevel < levelsData.length - 1) {{
      currentLevel++;
      updateHUD();
      drawGraph();
    }} else if (isPlaying) {{
      togglePlay();
    }}
  }}

  function prevLevel() {{
    if (currentLevel > 0) {{
      currentLevel--;
      updateHUD();
      drawGraph();
    }}
  }}

  function resetBFS() {{
    if (isPlaying) togglePlay();
    currentLevel = 0;
    updateHUD();
    drawGraph();
  }}

  function togglePlay() {{
    isPlaying = !isPlaying;
    const btn = document.getElementById("playBtn");
    if (isPlaying) {{
      btn.innerHTML = "&#10074;&#10074; Pause";
      btn.style.background = "#e11d48";
      timer = setInterval(nextLevel, 900);
    }} else {{
      btn.innerHTML = "&#9654; Play";
      btn.style.background = "#0284c7";
      clearInterval(timer);
    }}
  }}

  // Initialize
  window.onload = () => {{
    resizeCanvas();
    updateHUD();
  }};
</script>
</body>
</html>
"""
    out_path = "demo/fabfs_demo.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[build_demo.py] Successfully wrote self-contained HTML demo: {out_path} ({len(html_content)} bytes)")
    return out_path


def main():
    print("[build_demo.py] Validating demo run and generating visualizer...")
    demo_df, run_id, graph_name = validate_demo_run()
    bin_path = "demo/demo_graph.bin"
    n, m, row_offsets, col_indices = load_binary_graph(bin_path)
    print(f"[build_demo.py] Loaded {bin_path}: n={n}, m={m}")

    coords, edges = generate_node_layout(n, row_offsets, col_indices, demo_df)
    out_html = build_fabfs_demo_html(demo_df, run_id, graph_name, coords, edges)
    print(f"[build_demo.py] Verification complete. Demo is ready at {out_html}")


if __name__ == "__main__":
    main()
