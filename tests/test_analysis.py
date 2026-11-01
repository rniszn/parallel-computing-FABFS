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
"""

import os
import sys
import json
import tempfile
import unittest
import numpy as np
import pandas as pd

# Add analysis directory to module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis")))

from load import (
    CONTRACT_COLUMNS,
    check_header,
    load_runs_jsonl,
    check_input_hashes,
    check_gate1_thresholds_commit
)
from stats import (
    bootstrap_ci_median,
    run_pass_fail_check
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

        # 2. Failing fixture: Adaptive is 20% slower than min(Static, Dynamic)
        failing_data = pd.DataFrame([
            {"graph": "g1", "policy": "Adaptive", "median_ms": 120.0, "ci_lower_ms": 118.0, "ci_upper_ms": 122.0, "n_runs": 15},
            {"graph": "g1", "policy": "Pure-Static", "median_ms": 100.0, "ci_lower_ms": 98.0, "ci_upper_ms": 102.0, "n_runs": 15},
            {"graph": "g1", "policy": "Pure-Dynamic", "median_ms": 150.0, "ci_lower_ms": 145.0, "ci_upper_ms": 155.0, "n_runs": 15},
        ])
        passed_fail, res_fail = run_pass_fail_check(failing_data)
        self.assertFalse(passed_fail)
        self.assertFalse(res_fail.iloc[0]["passed"])

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
