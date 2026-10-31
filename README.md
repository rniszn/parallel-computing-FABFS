# FABFS: Frontier-Adaptive CPU-GPU Breadth-First Search

**Course Project:** IT-301 Parallel Computing  
**Target Hardware:** Ubuntu 22.04 LTS / Windows, NVIDIA RTX 4050 GPU (CUDA 12.x, Unified Memory), Multi-core x86_64 CPU  

---

## 📌 Project Overview

Breadth-First Search (BFS) is a fundamental graph traversal algorithm characterized by irregular memory access patterns and dramatic shifts in frontier characteristics across traversal levels. In power-law and real-world scale-free graphs, early and tail levels typically exhibit small, sparse frontiers where kernel launch overheads dominate, while mid-traversal levels exhibit massive frontiers with extreme degree variance and skewed neighbor lists.

Unlike traditional heterogeneous systems that lock in a static execution policy per graph run (e.g., pure CPU, pure GPU, or a fixed static partition), **FABFS (Frontier-Adaptive BFS)** makes dynamic, **per-level scheduling decisions** based on runtime frontier properties:
1. **Frontier Size ($|F_l|$)**: The total count of active vertices at level $l$.
2. **Degree Skew / Coefficient of Variation ($CV = \sigma / \mu$)**: The degree heterogeneity across frontier vertices, calculated via an online single-pass Welford algorithm.

---

## 🚀 Execution Modes

At each BFS level $l$, the adaptive decision engine selects one of three distinct execution paths:

| Mode | Target Hardware | Execution Strategy | Decision Trigger |
| :--- | :--- | :--- | :--- |
| **`SEQ`** | CPU (Single Thread) | Sequential traversal with zero synchronization or kernel launch overhead. | $\|F_l\| < T_{\min}$ |
| **`STATIC`** | CPU + GPU (Co-execution) | Work statically partitioned between CPU and GPU according to throughput ratio: $\frac{P_{\text{gpu}}}{P_{\text{cpu}} + P_{\text{gpu}}}$. | $CV \le T_{CV} \lor \|F_l\| \le T_{\text{par}}$ |
| **`DYNAMIC`** | CPU + GPU (Work-Stealing) | Fine-grained work-stealing using **Chase-Lev deques** with skew-adaptive chunk size $C_l = \text{clamp}\left(\frac{C_{\max}}{1 + \lambda \cdot CV}, C_{\min}, C_{\max}\right)$. | Skewed frontier ($CV > T_{CV}$) |

---

## 🏗️ Repository Layout & Module Architecture

```text
parallel-computing-FABFS/
├── CMakeLists.txt              # Unified CMake build system
├── scripts/
│   ├── env_check.sh            # M0: Environment and hardware verification
│   ├── fetch_graphs.sh         # M1: Automated download of SNAP evaluation datasets
│   ├── calibrate.cpp           # M6: Microbenchmark calibration harness
│   ├── run_matrix.sh           # M8: Automated 480-run evaluation matrix runner
│   └── run_small_test.sh       # Evaluator quick verification test (< 5 min)
├── src/
│   ├── common/
│   │   └── types.h             # Shared data contracts (Graph, Mode, LevelStats, LevelRecord)
│   ├── graph/                  # M1: SNAP parser, CSR builder, R-MAT generator, validator
│   ├── ref/                    # M2: Ground-truth reference serial CPU BFS
│   ├── cpu/                    # M3: OpenMP level-synchronous parallel BFS
│   ├── gpu/                    # M4: CUDA level-synchronous BFS (Unified Memory)
│   ├── sched/                  # M5: Chase-Lev deque & per-level work-stealing scheduler
│   ├── model/                  # M6: Welford statistics, cost model & decision gate
│   ├── log/                    # M7: Per-level telemetry CSV logger
│   └── bench/                  # M8: Benchmark runner with checkpoint & resume support
├── analysis/                   # M9: Statistical analysis, bootstrap CIs, and paper plots
├── tests/                      # Unit tests and correctness test suites for all modules
├── demo/                       # M10: Interactive offline visualizer (HTML5 / D3 / Canvas)
├── results/                    # Experimental CSV logs, machine env capture, and traces
└── report/                     # LaTeX source, final presentation, and generated figures
```

---

## 📐 Shared Data Contracts (`src/common/types.h`)

All modules adhere strictly to shared immutable types:
- `vid_t`: 32-bit unsigned vertex ID (`uint32_t`).
- `eid_t`: 64-bit unsigned edge offset (`uint64_t`).
- `Graph`: Symmetrized CSR format (`row_offsets` of size $n+1$, `col_indices` of size $m$).
- `LevelStats`: Per-level statistics ($|F_l|$, degree sum $W_l$, mean degree $\mu$, degree $CV$, and $d_{\max}$).
- `LevelRecord`: Level profiling records (chosen mode, chunk size, elapsed time $\mu s$, steal attempts/successes, GPU batches, correctness).

---

## 🛡️ Acceptance Gates & Quality Milestones

Development follows an incremental milestone gate system. A module cannot be merged until its acceptance gate passes:
- **Gate 1 (Correctness):** CPU (M3), GPU (M4), and Adaptive (M5) implementations must match the reference BFS (M2) with **zero distance mismatches** across all benchmark graphs.
- **Gate 2 (Determinism):** Identical graph inputs with identical seeds must produce bitwise identical BFS output distances.
- **Gate 3 (Calibration Frozen):** Thresholds ($T_{\min}$, $T_{CV}$, $T_{\text{par}}$) calibrated strictly on held-out graphs and committed to `config/thresholds.json` before evaluation.
- **Gate 4 (Complete Evaluation):** Full $4 \times 8 \times 15 = 480$ run experimental matrix completed with telemetry validation and `correct == true` on every run.

---

## 👥 Team & Responsibilities

- **Rohan G:** CUDA Runtime, Unified Memory management, pinned control plane, GPU proxy thread.
- **Narenderan S:** OpenMP parallel execution engine, Chase-Lev work-stealing deque, correctness suites.
- **Yash N Gandhi:** Online Welford frontier modeling, cost model derivation, decision engine.
- **Amogh R Gowda:** Graph ingestion pipelines, benchmarking harness, statistical analysis, and figures.
