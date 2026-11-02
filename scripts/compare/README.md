# FABFS Per-Graph Policy Comparison & Evaluation Scripts

This directory contains standalone terminal comparison scripts designed to evaluate BFS traversal under four execution policies (`SEQ`, `STATIC`, `DYNAMIC`, and `FABFS` Adaptive) plus an empirical `ORACLE` (per-level fastest mode upper bound).

Together, these scripts empirically answer the five core research questions of the project:
- **Q1:** Does the best strategy change across BFS levels?
- **Q2:** Does adaptive switching improve end-to-end traversal time?
- **Q3:** Can frontier topology ($|F|$, $\mu$, $CV$, $d_{\max}$) explain the selected strategy?
- **Q4:** How close is the model to the measured per-level oracle?
- **Q5:** Does behaviour generalise across graph types?

---

## 🚀 Running Comparisons per Graph

Each script is a standalone thin wrapper executing 2 warm-ups and $N$ randomized rounds (default $N=15$), producing aligned terminal **Tables A through E** and writing M7-compliant 28-column level rows.

### 1. California Road Network (`roadNet-CA`)
High-diameter planar road network (1.96M vertices, 555 levels).
```bash
# Standard 15-round evaluation
./scripts/compare/run_compare_roadnet_ca.sh

# Fast sanity evaluation (2 warmups + 3 rounds)
./scripts/compare/run_compare_roadnet_ca.sh --quick
```

### 2. Google Web Crawl Graph (`web-Google`)
Scale-free power-law web graph (875K vertices, 16 levels) exhibiting full phase transitions.
```bash
# Standard 15-round evaluation
./scripts/compare/run_compare_web_google.sh

# Fast sanity evaluation (2 warmups + 3 rounds)
./scripts/compare/run_compare_web_google.sh --quick
```

### 3. Synthetic Kronecker Graph (`rmat_a0.55`)
R-MAT Scale 20, $a=0.55$, edge factor 16, seed 12345 (1.04M vertices, 16.7M edges).
```bash
# Standard 15-round evaluation
./scripts/compare/run_compare_rmat20_a055.sh

# Fast sanity evaluation (2 warmups + 3 rounds)
./scripts/compare/run_compare_rmat20_a055.sh --quick
```

---

## 📊 Cross-Graph Aggregation (Answering Q5)

After running all three comparisons, run the aggregate script to display the cross-graph comparison matrix and evaluate the Q5 generalization verdict:
```bash
python3 scripts/compare/summarize_all.py
```

---

## 🛠️ CLI Options

All scripts accept the following standard flags:
- `--quick`: Quick evaluation with 2 unrecorded warmups and 3 timed rounds.
- `--rounds=<N>`: Number of timed rounds (default: 15).
- `--warmup=<W>`: Number of unrecorded warm-up rounds (default: 2).
- `--seed=<S>`: Seed for randomized policy execution order (default: 12345).
- `--threads=<T>`: CPU worker threads (default: 4).
- `--out-dir=<dir>`: Custom output directory (default: `results/compare/<graph>`).
- `--thresholds=<path>`: Path to frozen calibration thresholds (default: `config/thresholds.json`, read-only).
- `--help`: Print detailed help message.

---

## 📁 Output Artifacts

For each graph `<graph>`, results are stored in `results/compare/<graph>/`:
- `summary.json`: JSON machine-readable export of Tables A, C, D, E and question verdicts.
- `per_level.csv`: Per-level runtime and topological metrics across all BFS levels.
- `levels.csv`: Exact M7 28-column level rows with hardware and kernel profiling telemetry.
