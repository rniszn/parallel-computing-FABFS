#!/usr/bin/env python3
"""
M9 Unit Tests (tests/test_analysis.py)
Validates:
1. Header 28-column contract check (catches missing, extra, and misordered columns).
2. Run integrity check (rejects uncompleted runs, excludes timeouts).
3. Input hash verification (rejects unknown graphs).
4. Gate 1 vs frozen thresholds commit hash consistency.
5. Bootstrap confidence interval determinism with fixed seed.
6. Pass/fail criterion evaluation (positive and deliberately failing cases).
7. Figure regeneration consistency test (ensures reproducible outputs).
8. Missing policy gives NOT EVALUATED, not PASS.
9. Adaptive 4% slower than better fixed gives PASS, 6% slower gives FAIL.
10. Cell with n = 1 prints "n=1, no CI".
11. Script exits non-zero if any graph is NOT EVALUATED.
"""

import os
import sys
import json
import tempfile
import unittest
import subprocess
import numpy as np
import pandas as pd

# Add analysis directory to module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis")))

from load import (
    CONTRACT_COLUMNS,
    EVALUATION_GRAPHS,
    check_header,
    load_runs_jsonl,
    check_input_hashes,
    check_gate1_thresholds_commit
)
from stats import (
    bootstrap_ci_median,
    run_pass_fail_check,
    compute_overall_status
)
from tables import (
    generate_table2_policy_times,
    generate_table3_pass_fail
)
from figures import (
    plot_fig1_median_times,
    plot_fig3_mode_shares
)


class TestAnalysisPipeline(unittest.TestCase):

    def test_header_contract_success(self):
        """Header exactly matching the 28-column contract passes."""
        self.assertTrue(check_header(CONTRACT_COLUMNS))

    def test_header_contract_missing_column(self):
        """Header missing a column is rejected."""
        corrupted = [c for c in CONTRACT_COLUMNS if c != "gpu_idle_us"]
        with self.assertRaises(ValueError) as ctx:
            check_header(corrupted)
        self.assertIn("Missing", str(ctx.exception))

    def test_header_contract_extra_column(self):
        """Header with an unexpected column is rejected."""
        corrupted = CONTRACT_COLUMNS + ["extra_col"]
        with self.assertRaises(ValueError) as ctx:
            check_header(corrupted)
        self.assertIn("Extra", str(ctx.exception))

    def test_header_contract_order_mismatch(self):
        """Header with swapped columns is rejected."""
        corrupted = list(CONTRACT_COLUMNS)
        corrupted[0], corrupted[1] = corrupted[1], corrupted[0]
        with self.assertRaises(ValueError) as ctx:
            check_header(corrupted)
        self.assertIn("Column ordering mismatch", str(ctx.exception))

    def test_run_integrity_check(self):
        """Verify runs.jsonl parser categorizes complete vs timeout runs."""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl") as f:
            f.write(json.dumps({"run_id": "run_001", "status": "complete", "graph": "g1"}) + "\n")
            f.write(json.dumps({"run_id": "run_002", "status": "timeout", "graph": "g2"}) + "\n")
            tmp_path = f.name

        try:
            completed, timed_out, all_runs = load_runs_jsonl(tmp_path)
            self.assertIn("run_001", completed)
            self.assertNotIn("run_002", completed)
            self.assertIn("run_002", timed_out)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_input_hashes_check(self):
        """Graph absent from inputs.txt raises ValueError."""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as f:
            f.write("roadNet-CA 04b991d4d5285bda\n")
            f.write("web-Google 8c0f453f1eb1e24a\n")
            tmp_path = f.name

        try:
            self.assertTrue(check_input_hashes(["roadNet-CA", "web-Google"], tmp_path))
            with self.assertRaises(ValueError):
                check_input_hashes(["unknown-graph"], tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_gate1_thresholds_commit_match(self):
        """Mismatch between gate1.txt and thresholds.json commit raises ValueError."""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f_thresh:
            json.dump({"calibration_commit": "commit_AAA"}, f_thresh)
            t_path = f_thresh.name

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as f_gate1:
            f_gate1.write("# Git Commit: commit_BBB\n")
            g_path = f_gate1.name

        try:
            with self.assertRaises(ValueError) as ctx:
                check_gate1_thresholds_commit(g_path, t_path)
            self.assertIn("Commit mismatch", str(ctx.exception))
        finally:
            if os.path.exists(t_path):
                os.remove(t_path)
            if os.path.exists(g_path):
                os.remove(g_path)

    def test_bootstrap_ci_determinism(self):
        """Bootstrap CI with fixed seed produces identical bounds across runs."""
        sample_data = np.array([12.3, 14.1, 11.9, 13.5, 12.8, 14.0, 13.1, 12.5, 13.9, 12.2])
        med1, lo1, hi1 = bootstrap_ci_median(sample_data, num_resamples=1000, ci=95.0, seed=12345)
        med2, lo2, hi2 = bootstrap_ci_median(sample_data, num_resamples=1000, ci=95.0, seed=12345)

        self.assertEqual(med1, med2)
        self.assertEqual(lo1, lo2)
        self.assertEqual(hi1, hi2)

    def test_pass_fail_logic(self):
        """Verify pass/fail evaluation on passing and failing fixtures."""
        # 1. Passing fixture: Adaptive is within 5% of min(Static, Dynamic)
        passing_data = pd.DataFrame([
            {"graph": "g1", "policy": "Adaptive", "median_ms": 102.0, "ci_lower_ms": 100.0, "ci_upper_ms": 104.0, "n_runs": 15},
            {"graph": "g1", "policy": "Pure-Static", "median_ms": 100.0, "ci_lower_ms": 98.0, "ci_upper_ms": 102.0, "n_runs": 15},
            {"graph": "g1", "policy": "Pure-Dynamic", "median_ms": 150.0, "ci_lower_ms": 145.0, "ci_upper_ms": 155.0, "n_runs": 15},
        ])
        passed, res = run_pass_fail_check(passing_data)
        self.assertTrue(passed)
        self.assertTrue(res.iloc[0]["passed"])
        self.assertEqual(res.iloc[0]["status"], "PASS")

        # 2. Failing fixture: Adaptive is 20% slower than min(Static, Dynamic)
        failing_data = pd.DataFrame([
            {"graph": "g1", "policy": "Adaptive", "median_ms": 120.0, "ci_lower_ms": 118.0, "ci_upper_ms": 122.0, "n_runs": 15},
            {"graph": "g1", "policy": "Pure-Static", "median_ms": 100.0, "ci_lower_ms": 98.0, "ci_upper_ms": 102.0, "n_runs": 15},
            {"graph": "g1", "policy": "Pure-Dynamic", "median_ms": 150.0, "ci_lower_ms": 145.0, "ci_upper_ms": 155.0, "n_runs": 15},
        ])
        passed_fail, res_fail = run_pass_fail_check(failing_data)
        self.assertFalse(passed_fail)
        self.assertFalse(res_fail.iloc[0]["passed"])
        self.assertEqual(res_fail.iloc[0]["status"], "FAIL")

    def test_missing_policy_gives_not_evaluated(self):
        """Test 1: A graph with a missing policy gives NOT EVALUATED, not PASS."""
        # roadNet-CA fixture missing Pure-Dynamic
        partial_data = pd.DataFrame([
            {"graph": "roadNet-CA", "policy": "Adaptive", "median_ms": 1277.51, "ci_lower_ms": 1240.0, "ci_upper_ms": 1310.0, "n_runs": 1},
            {"graph": "roadNet-CA", "policy": "Pure-Static", "median_ms": 2771.70, "ci_lower_ms": 2700.0, "ci_upper_ms": 2850.0, "n_runs": 1},
            # Pure-Dynamic is missing
        ])
        overall_pass, res = run_pass_fail_check(partial_data)
        self.assertFalse(overall_pass, "Overall pass must be False if any graph is NOT EVALUATED")
        self.assertEqual(res.iloc[0]["status"], "NOT EVALUATED")
        self.assertNotEqual(res.iloc[0]["status"], "PASS")

    def test_threshold_ratio_boundary(self):
        """Test 2: Adaptive 4% slower than the better fixed policy gives PASS. 6% slower gives FAIL."""
        # 4% slower: 104.0 vs 100.0 (ratio 1.04 <= 1.05) -> PASS
        pass_data = pd.DataFrame([
            {"graph": "g1", "policy": "Adaptive", "median_ms": 104.0, "ci_lower_ms": 102.0, "ci_upper_ms": 106.0, "n_runs": 15},
            {"graph": "g1", "policy": "Pure-Static", "median_ms": 100.0, "ci_lower_ms": 98.0, "ci_upper_ms": 102.0, "n_runs": 15},
            {"graph": "g1", "policy": "Pure-Dynamic", "median_ms": 200.0, "ci_lower_ms": 195.0, "ci_upper_ms": 205.0, "n_runs": 15},
        ])
        overall_p, res_p = run_pass_fail_check(pass_data)
        self.assertTrue(overall_p)
        self.assertEqual(res_p.iloc[0]["status"], "PASS")

        # 6% slower: 106.0 vs 100.0 (ratio 1.06 > 1.05) -> FAIL
        fail_data = pd.DataFrame([
            {"graph": "g1", "policy": "Adaptive", "median_ms": 106.0, "ci_lower_ms": 104.0, "ci_upper_ms": 108.0, "n_runs": 15},
            {"graph": "g1", "policy": "Pure-Static", "median_ms": 100.0, "ci_lower_ms": 98.0, "ci_upper_ms": 102.0, "n_runs": 15},
            {"graph": "g1", "policy": "Pure-Dynamic", "median_ms": 200.0, "ci_lower_ms": 195.0, "ci_upper_ms": 205.0, "n_runs": 15},
        ])
        overall_f, res_f = run_pass_fail_check(fail_data)
        self.assertFalse(overall_f)
        self.assertEqual(res_f.iloc[0]["status"], "FAIL")

    def test_single_run_no_ci(self):
        """Test 3: A cell with n = 1 prints 'n=1, no CI'."""
        # 1. bootstrap_ci_median on single sample returns NaN for bounds
        med, lo, hi = bootstrap_ci_median(np.array([123.45]))
        self.assertEqual(med, 123.45)
        self.assertTrue(np.isnan(lo))
        self.assertTrue(np.isnan(hi))

        # 2. generate_table2_policy_times formats n=1 cell with 'n=1, no CI'
        single_run_stats = pd.DataFrame([
            {"graph": "roadNet-CA", "policy": "Adaptive", "median_ms": 1277.51, "ci_lower_ms": np.nan, "ci_upper_ms": np.nan, "n_runs": 1}
        ])
        with tempfile.TemporaryDirectory() as tmp_dir:
            t2 = generate_table2_policy_times(single_run_stats, out_dir=tmp_dir)
            cell_val = t2.iloc[0]["Adaptive"]
            self.assertIn("n=1, no CI", cell_val)
            self.assertIn("1277.51 (n=1)", cell_val)

    def test_script_exits_nonzero_on_not_evaluated(self):
        """Test 4: The script exits non-zero if any graph is NOT EVALUATED."""
        stats_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis", "stats.py"))
        res = subprocess.run([sys.executable, stats_script], capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0, "stats.py must exit non-zero when evaluation graphs are NOT EVALUATED or FAIL")
        self.assertIn("NOT EVALUATED", res.stdout)
        self.assertIn("OVERALL STATUS: INCOMPLETE", res.stdout)

    def test_fixture_graph_excluded_from_evaluation(self):
        """Test: A fixture graph outside the 8 evaluation graphs does not appear in table3 and does not change overall status."""
        # DataFrame with an evaluation graph + fixture graph rmat10_seed1
        df_with_fixture = pd.DataFrame([
            {"graph": "roadNet-CA", "policy": "Adaptive", "median_ms": 100.0, "ci_lower_ms": 95.0, "ci_upper_ms": 105.0, "n_runs": 15},
            {"graph": "roadNet-CA", "policy": "Pure-Static", "median_ms": 100.0, "ci_lower_ms": 95.0, "ci_upper_ms": 105.0, "n_runs": 15},
            {"graph": "roadNet-CA", "policy": "Pure-Dynamic", "median_ms": 100.0, "ci_lower_ms": 95.0, "ci_upper_ms": 105.0, "n_runs": 15},
            {"graph": "rmat10_seed1", "policy": "Adaptive", "median_ms": 0.5, "ci_lower_ms": 0.4, "ci_upper_ms": 0.6, "n_runs": 1},
            {"graph": "extra_fixture", "policy": "Adaptive", "median_ms": 0.5, "ci_lower_ms": 0.4, "ci_upper_ms": 0.6, "n_runs": 1},
        ])

        status, pf_df = run_pass_fail_check(df_with_fixture)
        self.assertNotIn("rmat10_seed1", pf_df["graph"].values)
        self.assertNotIn("extra_fixture", pf_df["graph"].values)
        for g in pf_df["graph"].values:
            self.assertIn(g, EVALUATION_GRAPHS)

        with tempfile.TemporaryDirectory() as tmp_dir:
            t3 = generate_table3_pass_fail(pf_df, out_dir=tmp_dir)
            self.assertNotIn("rmat10_seed1", t3["Graph"].values)
            self.assertNotIn("extra_fixture", t3["Graph"].values)

    def test_three_status_outcomes(self):
        """Test: Verify all three overall status outcomes: INCOMPLETE, FAIL, PASS."""
        # 1. INCOMPLETE: at least one evaluation graph is NOT EVALUATED
        incomplete_df = pd.DataFrame([
            {"graph": "g1", "status": "PASS"},
            {"graph": "g2", "status": "NOT EVALUATED"},
        ])
        status_inc = compute_overall_status(incomplete_df)
        self.assertEqual(status_inc, "INCOMPLETE")
        self.assertFalse(status_inc)

        # 2. FAIL: every graph is evaluated, but at least one FAILs
        fail_df = pd.DataFrame([
            {"graph": "g1", "status": "PASS"},
            {"graph": "g2", "status": "FAIL"},
        ])
        status_fail = compute_overall_status(fail_df)
        self.assertEqual(status_fail, "FAIL")
        self.assertFalse(status_fail)

        # 3. PASS: every graph is evaluated and all PASS
        pass_df = pd.DataFrame([
            {"graph": "g1", "status": "PASS"},
            {"graph": "g2", "status": "PASS"},
        ])
        status_pass = compute_overall_status(pass_df)
        self.assertEqual(status_pass, "PASS")
        self.assertTrue(status_pass)

    def test_figure_regeneration_consistency(self):
        """Regenerating a figure from identical data produces exact duplicate data output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_stats = pd.DataFrame([
                {"graph": "roadNet-CA", "policy": "Adaptive", "median_ms": 50.0, "ci_lower_ms": 48.0, "ci_upper_ms": 52.0},
                {"graph": "roadNet-CA", "policy": "Pure-Static", "median_ms": 55.0, "ci_lower_ms": 53.0, "ci_upper_ms": 57.0},
                {"graph": "roadNet-CA", "policy": "Pure-Dynamic", "median_ms": 60.0, "ci_lower_ms": 58.0, "ci_upper_ms": 62.0},
                {"graph": "roadNet-CA", "policy": "Oracle-Replay", "median_ms": 49.0, "ci_lower_ms": 47.0, "ci_upper_ms": 51.0},
            ])

            # Run 1
            _, _, csv1 = plot_fig1_median_times(test_stats, out_dir=tmp_dir)
            with open(csv1, "r", encoding="utf-8") as f:
                content1 = f.read()

            # Run 2
            _, _, csv2 = plot_fig1_median_times(test_stats, out_dir=tmp_dir)
            with open(csv2, "r", encoding="utf-8") as f:
                content2 = f.read()

            self.assertEqual(content1, content2)


if __name__ == "__main__":
    unittest.main()
