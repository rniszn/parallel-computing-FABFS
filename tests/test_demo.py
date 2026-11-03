#!/usr/bin/env python3
"""
Unit tests for FABFS evaluator demo visualizer (demo/fabfs_demo.html & demo/build_demo.py).
Verifies data integrity checks, 100% offline self-containment, UI elements,
mode color definitions, beside-graph telemetry, controls, and mobile responsive rules.
"""

import os
import sys
import json
import re
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo")))
from build_demo import validate_demo_run, build_fabfs_demo_html


class TestFABFSDemo(unittest.TestCase):
    def setUp(self):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.html_path = os.path.join(self.repo_root, "demo", "fabfs_demo.html")
        self.demo_run_path = os.path.join(self.repo_root, "demo", "demo_run.csv")
        self.levels_path = os.path.join(self.repo_root, "results", "levels.csv")
        self.thresholds_path = os.path.join(self.repo_root, "config", "thresholds.json")

    def test_demo_file_exists_and_offline(self):
        """Verifies demo/fabfs_demo.html exists, is non-empty, and contains zero external HTTP/HTTPS requests."""
        self.assertTrue(os.path.exists(self.html_path), "demo/fabfs_demo.html must exist")
        with open(self.html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertGreater(len(content), 10000, "demo/fabfs_demo.html must be populated")
        
        # Verify completely offline (no http:// or https://)
        external_urls = re.findall(r'https?://[^\s"\'<>]+', content)
        self.assertEqual(len(external_urls), 0, f"External network URLs found in demo: {external_urls}")
        self.assertNotIn("<script src=\"http", content)
        self.assertNotIn("<link rel=\"stylesheet\" href=\"http", content)

    def test_demo_refuses_on_mismatched_run(self):
        """Verifies that validate_demo_run refuses to build if demo_run.csv does not match levels.csv."""
        # Baseline should validate cleanly
        demo_df, run_id, graph_name = validate_demo_run()
        self.assertGreater(len(demo_df), 0)
        self.assertEqual(graph_name, "rmat10_seed1")

        # Test with non-existent levels file
        with self.assertRaises((FileNotFoundError, ValueError)):
            validate_demo_run(levels_csv_path="non_existent_levels.csv")

    def test_persistent_walkthrough_badge(self):
        """Verifies that the demo has a persistent walkthrough / real trace badge."""
        with open(self.html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn('id="traceTypeBadge"', content, "Badge element id traceTypeBadge must exist")
        self.assertIn("Synthetic walkthrough", content, "Default badge text must be Synthetic walkthrough")
        self.assertIn("Real trace", content, "Dynamic trace text Real trace must be supported in JS")

    def test_beside_graph_telemetry(self):
        """Verifies that frontier size, mode chosen, chunk count (DYNAMIC) and CPU/GPU time are displayed beside the graph."""
        with open(self.html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn('id="canvasLiveStats"', content, "Beside-graph stats overlay must exist")
        self.assertIn('id="liveFrontier"', content, "Frontier size element must exist beside graph")
        self.assertIn('id="liveMode"', content, "Mode chosen element must exist beside graph")
        self.assertIn('id="liveChunks"', content, "Chunk count element must exist beside graph")
        self.assertIn('id="liveDeviceTimes"', content, "CPU/GPU times element must exist beside graph")
        self.assertIn('id="liveTotalTime"', content, "Total level time element must exist beside graph")

    def test_mode_legend_and_colours(self):
        """Verifies legend for the three modes with matching colors (SEQ #38bdf8, STATIC #4ade80, DYNAMIC #fb923c)."""
        with open(self.html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("#38bdf8", content, "SEQ color #38bdf8 must be defined")
        self.assertIn("#4ade80", content, "STATIC color #4ade80 must be defined")
        self.assertIn("#fb923c", content, "DYNAMIC color #fb923c must be defined")
        self.assertIn("canvas-mode-legend", content, "Canvas mode legend toolbar must exist")

    def test_controls_shortcuts_and_step_counter(self):
        """Verifies controls: previous, next, play, reset, keyboard shortcuts, and visible counter 'Level X of Y'."""
        with open(self.html_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Controls buttons
        self.assertIn('id="prevBtn"', content, "Previous level button must exist")
        self.assertIn('id="nextBtn"', content, "Next level button must exist")
        self.assertIn('id="playBtn"', content, "Play/Pause button must exist")
        self.assertIn('id="resetBtn"', content, "Reset button must exist")

        # Step counter
        self.assertIn('id="levelIndicator"', content, "Step counter element must exist")
        self.assertIn("Level 1 of 4", content, "Default step counter must be formatted as Level X of Y")

        # Keyboard shortcuts
        self.assertIn("ArrowLeft", content, "Left arrow keyboard shortcut must be handled")
        self.assertIn("ArrowRight", content, "Right arrow keyboard shortcut must be handled")
        self.assertIn("Space", content, "Space keyboard shortcut must be handled")

    def test_mode_meaning_under_graph(self):
        """Verifies one-sentence explanation under the graph describing what the current level's mode means."""
        with open(self.html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn('id="modeMeaningBar"', content, "Mode meaning bar under graph must exist")
        self.assertIn('id="modeMeaningText"', content, "Mode meaning text element must exist")
        self.assertIn("SEQ Mode: A single CPU core processes", content)

    def test_mobile_responsive_layout(self):
        """Verifies mobile phone layout support: controls at least 44px tall, text at least 14px, no horizontal scroll."""
        with open(self.html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("@media (max-width: 900px)", content)
        self.assertIn("@media (max-width: 480px)", content)
        self.assertIn("min-height: 44px", content, "Touch targets must be at least 44px tall")
        self.assertIn("overflow-x: hidden", content, "Must prevent horizontal scroll")

    def test_footer_calibration_and_commit(self):
        """Verifies that thresholds and calibration commit hash from config/thresholds.json are displayed in footer."""
        with open(self.thresholds_path, "r", encoding="utf-8") as f:
            thresh = json.load(f)

        with open(self.html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn('class="app-footer"', content, "Footer element must exist")
        self.assertIn(thresh["calibration_commit"], content, "Calibration commit hash must appear in footer")
        self.assertIn(str(thresh["T_min"]), content, "T_min must appear in footer")
        self.assertIn(str(thresh["T_CV"]), content, "T_CV must appear in footer")
        self.assertIn(str(thresh["T_par"]), content, "T_par must appear in footer")
        self.assertIn(str(thresh["alpha"]), content, "alpha must appear in footer")


if __name__ == "__main__":
    unittest.main()
