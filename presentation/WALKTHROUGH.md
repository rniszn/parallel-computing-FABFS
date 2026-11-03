# FABFS Evaluator Walk-Through Script

**Project:** FABFS (Frontier-Adaptive CPU-GPU BFS)  
**Course:** IT-301 Parallel Computing, NITK Surathkal  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, Ada Lovelace compute capability 8.9) + 13th Gen Intel Core i5-13420H (Ubuntu 22.04 LTS / WSL2)

---

### Step 1: Open the Interactive Demo & Demonstrate Mode Transitions
1. **Open the Demo:**
   - In any web browser, open the self-contained visualizer: [`demo/fabfs_demo.html`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/demo/fabfs_demo.html).
   - Point out that this is **not a mock-up**: it directly replays the exact per-level telemetry rows produced by the M7 CSV logger (`results/levels.csv`).
   - Note the **persistent badge** in the header at all times:
     - Shows **`⚙ SYNTHETIC WALKTHROUGH`** when viewing the 1K-node 2D concentric graph (`rmat10_seed1`).
     - Automatically updates to **`► REAL TRACE`** when viewing SNAP evaluation graphs (`web-Google`, `com-lj`, `roadNet-CA`).
   - Click the prominent **`? Research Questions (5/5)`** button in the header:
     - Open the interactive modal to introduce the **5 core project hypotheses**:
       1. *Does the best strategy change across BFS levels?* (Q1)
       2. *Does adaptive switching improve end-to-end traversal time?* (Q2)
       3. *Can frontier topology explain the selected strategy?* (Q3)
       4. *How close is the model to the measured per-level oracle?* (Q4)
       5. *Does behaviour generalise across graph types?* (Q5)
     - Note that each card has an interactive jump button that instantly scrubs the visualizer to that specific live empirical proof!
   - Mention the graph selector dropdown in the header, supporting **4 distinct graph topologies**:
     - **`rmat10_seed1`** (1,024 nodes): 2D concentric ring layout showing individual node activations and frontier pulses. Verified by SHA256 (`f08f5230...`).
     - **`web-Google`** (875,713 nodes, run `run_062`): Real web crawl displaying full phase transitions (**`SEQ` $\rightarrow$ `STATIC` $\rightarrow$ `DYNAMIC` $\rightarrow$ `STATIC` $\rightarrow$ `SEQ`**).
     - **`com-lj`** (3,997,962 nodes, run `run_126`): Massive LiveJournal social network exhibiting extreme degree skew ($CV > 2.0$) and Chase-Lev work-stealing chunks.
     - **`roadNet-CA`** (1,965,206 nodes, run `run_005`): High-diameter planar California road network across 555 levels, showing how sequential dispatch prevents GPU launch thrashing on thin frontiers.

2. **Step Through the Traversal & Point to Mode Decisions:**
   - **Controls & Step Counter:**
     - Point out the **clear control buttons**: `◀ Prev`, `▶ Play` / `⏸ Pause`, `Next ▶`, and `↺ Reset`.
     - Highlight the visible step counter: e.g., **`Level 1 of 4`** on `rmat10_seed1` and **`Level 3 of 17`** (or `Level 4 of 16`) on `web-Google`.
     - Demonstrate the **keyboard shortcuts**:
       - Press **Left Arrow (`◀`)** to step back.
       - Press **Right Arrow (`▶`)** to step forward.
       - Press **Space** to toggle automated playback.
   - **Beside-Graph Active Telemetry Card:**
     - Direct the evaluator's attention to the active level telemetry card located **directly next to the graph canvas**:
       - **Frontier Size $|F_l|$**: Active vertex count for the current level.
       - **Mode Chosen**: Color-coded badge (`SEQ` blue, `STATIC` green, `DYNAMIC` orange).
       - **Chunk Count (DYNAMIC)**: Displays dynamic chunk size $C_l$ and total chunks (e.g., `128 chunks (C_l=4,096)`) when in `DYNAMIC` mode, or `N/A` when in `SEQ` or `STATIC`.
       - **CPU / GPU Time**: Per-device expansion duration in $\mu\text{s}$.
       - **Total Level Time**: Complete traversal time for this level in $\mu\text{s}$.
   - **One-Sentence Mode Meaning Bar Under Graph:**
     - Point out the dedicated banner directly under the graph canvas stating what the current level's mode means in plain English:
       - *SEQ Mode:* "A single CPU core processes the frontier in L1/L2 cache without thread synchronization or GPU kernel launch latency."
       - *STATIC Mode:* "Frontier edges are statically partitioned between CPU OpenMP threads and the GPU with deterministic dual-device load balancing."
       - *DYNAMIC Mode:* "CPU threads and GPU execute lock-free Chase-Lev deques with work-stealing to eliminate load imbalance on skewed hub nodes."
   - **Step-by-step Demonstration on `web-Google`:**
     - Select **`web-Google`** from the dropdown:
       - **Levels 0–2 ($|F| \le 422 < 1024$):** Dispatched in **`SEQ MODE`** (cyan #38bdf8).
       - **Level 3 ($|F| = 4,468 \le 8,192$):** Dispatched in **`STATIC MODE`** (green #4ade80), splitting edges between CPU and GPU ($\alpha = 15.3\%$).
       - **Levels 4–9 ($|F| = 21,226$ to $310,197 > 8,192$ and $CV = 1.62 \text{ to } 6.54 > 1.5$):** Dispatched in **`DYNAMIC MODE`** (orange #fb923c) with Chase-Lev work-stealing and dynamic chunking ($C_l = 4,096$).
       - **Levels 10–11 ($|F| \le 6,704 \le 8,192$):** Automatically switches back to **`STATIC MODE`** as the frontier recedes.
       - **Levels 12–15 ($|F| \le 529 < 1,024$):** Switches back to **`SEQ MODE`** to finish the trailing tail with zero parallel overhead.
   - **Footer Verification:**
     - Point out the footer displaying frozen calibration thresholds ($T_{\text{min}}=1024, T_{\text{CV}}=1.5, T_{\text{par}}=8192, \alpha=0.1526, L=39.99~\mu\text{s}, C_{\text{steal}}=135.6~\text{ns}$) and calibration commit hash (`8dd8ad73...`), loaded from `config/thresholds.json`.

*(Fallback: If browser display is unavailable, switch to the static visual steps in [`presentation/fallback/`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/presentation/fallback)).*

---

### Step 2: Show Quality Gates & Benchmark Matrix (Slide 5)
1. **Open Slide 5:**
   - Present Slide 5 of [`presentation/FABFS.pptx`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/presentation/FABFS.pptx):
2. **Key Explanations for Evaluator:**
   - **Quality Gate 1 (Baseline Correctness & Calibration): PASSED (100% Correct)**
     - 100% distance verification against M2 reference BFS across all thread counts ($T \in \{1,2,4,8\}$) and graphs (`results/gate1.txt`).
     - Thresholds frozen in `config/thresholds.json` before evaluation began; zero calibration data contamination.
   - **Quality Gate 4 (Matrix Evaluation & Acceptance): NOT EVALUATED / PENDING FULL RUN**
     - Protocol: 480 target runs (4 policies x 8 evaluation graphs x 15 rounds).
     - Status: 25 of 480 runs recorded in current repository state. Because comparison baseline runs (such as Pure-Dynamic on `roadNet-CA`) are incomplete, pass/fail is NOT EVALUATED. All completed levels maintain `correct = true`.
     - Full matrix execution is pending on a clean native Ubuntu machine.

---

### Step 3: Show Figure 1 (Median Traversal Time per Policy, Slide 6)
1. **Open Figure 1:**
   - Present [`presentation/figures/fig1_median_times.png`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/presentation/figures/fig1_median_times.png) (or Slide 6 of [`presentation/FABFS.pptx`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/presentation/FABFS.pptx)).
2. **Key Explanations for Evaluator:**
   - Shows the median total execution time across the 4 policies: **Adaptive** (blue #38bdf8), **Pure-Static** (green #4ade80), **Pure-Dynamic** (orange #fb923c), and **Oracle-Replay** (purple).
   - Whiskers display the 95% bootstrap confidence intervals (cells with $n=1$ show "n=1, no CI").
   - Point out available recorded medians:
     - On planar road networks (`roadNet-CA`), Adaptive achieves $1,277.51~\text{ms}$ ($n=1$), while Pure-Static is $2,771.70~\text{ms}$ ($n=2$).
     - On power-law graphs (`web-Google`), Adaptive achieves $1,593.36~\text{ms}$ ($n=1$), while Pure-Dynamic is $1,751.72~\text{ms}$ ($n=1$).

---

### Step 4: Show the Formal Pass/Fail Acceptance Table (Slide 11)
1. **Open Table 3:**
   - Present [`report/tables/table3_pass_fail.csv`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/report/tables/table3_pass_fail.csv) (or Slide 11 of [`presentation/FABFS.pptx`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/presentation/FABFS.pptx)).
2. **Evaluation Criterion & Verification:**
   - **Criterion:** On every evaluation graph, Adaptive median total time is within 5% of the better of Pure-Static and Pure-Dynamic:
     $$T_{\text{adaptive}} \le 1.05 \times \min(T_{\text{static}}, T_{\text{dynamic}})$$
   - Point out that **Pass/fail is not yet evaluated: the full 480-run matrix is pending**:
     - On `roadNet-CA`: Pure-Dynamic has no recorded run in the partial matrix, so the 5% rule cannot yet be evaluated. Table 3 strictly marks it as **`NOT EVALUATED`** (never false PASS).
     - On `web-Google`, `com-lj`, and R-MAT graphs: missing baseline cells also produce **`NOT EVALUATED`**.
   - Emphasize that **all numbers and statuses are automatically produced** from `results/levels.csv` by `analysis/stats.py` with zero manual intervention or hand-tuned values.

---

### Step 5: Show the Frozen Calibration Thresholds & Commit Hash
1. **Open Thresholds Configuration:**
   - Open [`config/thresholds.json`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/config/thresholds.json) and [`results/gate1.txt`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/results/gate1.txt).
2. **Show the Frozen Calibration Provenance:**
   - **Calibration Commit Hash:** `8dd8ad730e2eb6c5bf6e9491eb8401f9c7d792e3` (exact match between Gate 1 record, footer of the visualizer demo, and `thresholds.json`).
   - **Frozen Parameters:**
     - $T_{\text{min}} = 1024$ vertices (Sequential threshold)
     - $T_{\text{CV}} = 1.5$ (Degree skew threshold for work stealing)
     - $T_{\text{par}} = 8192$ vertices (Parallel static cutoff)
     - $\alpha = 0.1526$ (Static GPU work share $= P_{\text{gpu}} / (P_{\text{cpu}} + P_{\text{gpu}})$)
     - $C_{\text{max}} = 4096$, $C_{\text{min}} = 32$, $\lambda = 1.5$ (Dynamic chunk clamping)
     - $\text{GPU\_MIN\_BATCH} = 4096$ vertices (Coalescing minimum for GPU proxy)
     - $C_{\text{steal}} = 135.6~\text{ns}$, $L_{\text{launch}} = 39.99~\mu\text{s}$
3. **Closing Remark:**
   - Reiterate that calibration was performed strictly on calibration graphs (`roadNet-PA`, `com-Amazon`, `com-DBLP`, `rmat_s16`) and frozen before evaluation began.
   - All modules M0 through M10 are fully tested, self-contained, and reproducible via one command: `./analysis/run_all.sh`.
