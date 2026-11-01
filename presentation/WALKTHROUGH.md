# FABFS Evaluator Walk-Through Script

**Project:** FABFS (Frontier-Adaptive CPU-GPU BFS)  
**Course:** IT-301 Parallel Computing, NITK Surathkal  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, Ada Lovelace compute capability 8.9) + 13th Gen Intel Core i5-13420H (Ubuntu 22.04 LTS / WSL2)

---

### Step 1: Open the Interactive Demo & Demonstrate Mode Transitions
1. **Open the Demo:**
   - In any web browser, open the self-contained visualizer: [`demo/fabfs_demo.html`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/demo/fabfs_demo.html).
   - Point out that this is **not a mock-up**: it directly replays the exact per-level telemetry rows produced by the M7 CSV logger (`results/levels.csv`, run `demo_rmat10_run1`).
   - Mention the graph: R-MAT Scale 10 (`rmat10_seed1`), completely disjoint from the calibration and evaluation suites, verified by SHA256 (`f08f5230...`).

2. **Step Through the Traversal & Point to Mode Decisions:**
   - Click **Next ▶** or press **Play**:
     - **Level 0 ($|F|=1$):** Point to the **`SEQ MODE`** indicator. Because $|F_0| = 1 < T_{\text{min}} = 1024$, the decision engine dispatches strictly to a single CPU thread, eliminating kernel launch and thread pool synchronization overhead.
     - **Boundary Transition & Larger Graphs:** Explain the Algorithm 1 rule:
       $$\text{Mode}(F_l) = \begin{cases} 
       \text{SEQ} & \text{if } |F_l| < T_{\text{min}} \\
       \text{STATIC} & \text{else if } CV \le T_{\text{CV}} \text{ or } |F_l| \le T_{\text{par}} \\
       \text{DYNAMIC} & \text{otherwise}
       \end{cases}$$
     - In dense / skewed levels (as observed on `web-Google` and `com-Orkut`), as the frontier expands beyond $T_{\text{min}}$, the policy transitions smoothly from **`SEQ` $\rightarrow$ `STATIC` $\rightarrow$ `DYNAMIC` $\rightarrow$ `STATIC` $\rightarrow$ `SEQ`**.
     - Point out the work-stealing metrics on the HUD: dynamic chunk size $C_l$, steal attempts vs successes, and GPU proxy idle time ($\approx 0~\mu\text{s}$).

*(Fallback: If browser display is unavailable, switch to the static visual steps in [`presentation/fallback/`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/presentation/fallback)).*

---

### Step 2: Show Figure 1 (Median Traversal Time per Policy)
1. **Open Figure 1:**
   - Present [`presentation/figures/fig1_median_times.png`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/presentation/figures/fig1_median_times.png) (or slide 6 of [`presentation/FABFS.pptx`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/presentation/FABFS.pptx)).
2. **Key Explanations for Evaluator:**
   - Shows the median total execution time across the 4 policies: **Adaptive** (blue), **Pure-Static** (green), **Pure-Dynamic** (coral/red), and **Oracle-Replay** (purple).
   - Whiskers display the 95% bootstrap confidence intervals (1,000 resamples, fixed seed `12345`).
   - Point out that **Adaptive consistently tracks or outperforms the best fixed policy**:
     - On planar road networks (`roadNet-CA`), Adaptive achieves $1,277.51~\text{ms}$, avoiding the multi-second GPU sync penalties of Pure-Static ($2,771.70~\text{ms}$).
     - On power-law graphs (`web-Google`), Adaptive achieves $1,593.36~\text{ms}$, outperforming Pure-Dynamic ($1,751.72~\text{ms}$).

---

### Step 3: Show the Formal Pass/Fail Acceptance Table
1. **Open Table 3:**
   - Present [`report/tables/table3_pass_fail.csv`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/report/tables/table3_pass_fail.csv) (or [`report/tables/table3_pass_fail.tex`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/report/tables/table3_pass_fail.tex) / Slide 11).
2. **Evaluation Criterion & Verification:**
   - **Criterion:** On every evaluation graph, Adaptive median total time is within 5% of the better of Pure-Static and Pure-Dynamic:
     $$T_{\text{adaptive}} \le 1.05 \times \min(T_{\text{static}}, T_{\text{dynamic}})$$
   - Point out the **`roadNet-CA`** row:
     - Adaptive: $1,277.51~\text{ms}$
     - Best Fixed (Pure-Static): $2,771.70~\text{ms}$
     - Relative Difference: **$-53.91\%$** $\rightarrow$ **PASS**!
   - Emphasize that **all numbers are automatically produced** from `results/levels.csv` by `analysis/stats.py` with zero manual intervention or hand-tuned values.

---

### Step 4: Show the Frozen Calibration Thresholds & Commit Hash
1. **Open Thresholds Configuration:**
   - Open [`config/thresholds.json`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/config/thresholds.json) and [`results/gate1.txt`](file:///c:/Users/rohan/Desktop/SEM%205/parallel-computing-FABFS/results/gate1.txt).
2. **Show the Frozen Calibration Provenance:**
   - **Calibration Commit Hash:** `8dd8ad730e2eb6c5bf6e9491eb8401f9c7d792e3` (exact match between Gate 1 record and `thresholds.json`).
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
