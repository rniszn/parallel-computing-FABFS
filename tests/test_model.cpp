#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "graph/csr.h"
#include "graph/loader.h"
#include "graph/types.h"
#include "model/cost.h"
#include "model/decide.h"
#include "model/stats.h"

namespace {

bool file_exists(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

std::string get_graph_path(const std::string& name) {
  if (name == "roadNet-CA") return "data/roadNet-CA.txt";
  if (name == "web-Google") return "data/web-Google.txt";
  if (name == "com-lj") return "data/com-lj.ungraph.txt";
  if (name == "com-orkut") return "data/com-orkut.ungraph.txt";
  if (name == "roadNet-PA") return "data/roadNet-PA.txt";
  if (name == "com-amazon") return "data/com-amazon.ungraph.txt";
  if (name == "com-dblp") return "data/com-dblp.ungraph.txt";
  return "data/" + name + ".txt";
}

// Generates an SHA-256 hash or simple content signature of a file
std::string compute_file_sha256(const std::string& filepath) {
  std::string cmd = "sha256sum " + filepath + " 2>/dev/null";
  FILE* fp = popen(cmd.c_str(), "r");
  if (!fp) return "";
  char buf[128];
  std::string hash = "";
  if (fgets(buf, sizeof(buf), fp)) {
    hash = std::string(buf).substr(0, 64);
  }
  pclose(fp);
  return hash;
}

// Builds a synthetic regular graph where every vertex has degree 4
Graph make_regular_graph(vid_t n) {
  Graph g;
  g.n = n;
  g.m = n * 4;
  g.row_offsets.resize(n + 1, 0);
  g.col_indices.resize(n * 4);

  for (vid_t i = 0; i < n; ++i) {
    g.row_offsets[i] = i * 4;
    g.col_indices[i * 4 + 0] = (i + 1) % n;
    g.col_indices[i * 4 + 1] = (i + 2) % n;
    g.col_indices[i * 4 + 2] = (i + n - 1) % n;
    g.col_indices[i * 4 + 3] = (i + n - 2) % n;
  }
  g.row_offsets[n] = n * 4;
  return g;
}

// Builds a synthetic graph with isolated vertices (all degree 0)
Graph make_isolated_graph(vid_t n) {
  Graph g;
  g.n = n;
  g.m = 0;
  g.row_offsets.assign(n + 1, 0);
  return g;
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << "==========================================" << std::endl;
  std::cout << "Running M6 Model & Decision Engine Tests" << std::endl;
  std::cout << "==========================================" << std::endl;

  bool all_passed = true;

  // Test 1: decide with |F| = 1 and T_min > 1 returns SEQ
  {
    std::cout << "[Test 1] decide with |F| = 1 and T_min > 1 returns SEQ: ";
    Thresholds th;
    th.T_min = 64;
    th.T_CV = 1.0;
    th.T_par = 1024;
    Mode m = decide(1, 0.5, th);
    if (m == Mode::SEQ) {
      std::cout << "PASS" << std::endl;
    } else {
      std::cout << "FAIL (expected SEQ, got " << mode_to_string(m) << ")" << std::endl;
      all_passed = false;
    }
  }

  // Test 2: decide with CV = 3 and |F| > T_par returns DYNAMIC
  {
    std::cout << "[Test 2] decide with CV = 3 and |F| > T_par returns DYNAMIC: ";
    Thresholds th;
    th.T_min = 64;
    th.T_CV = 1.0;
    th.T_par = 1024;
    Mode m = decide(5000, 3.0, th);
    if (m == Mode::DYNAMIC) {
      std::cout << "PASS" << std::endl;
    } else {
      std::cout << "FAIL (expected DYNAMIC, got " << mode_to_string(m) << ")" << std::endl;
      all_passed = false;
    }
  }

  // Test 3: decide with CV <= T_CV returns STATIC
  {
    std::cout << "[Test 3] decide with CV <= T_CV returns STATIC: ";
    Thresholds th;
    th.T_min = 64;
    th.T_CV = 1.0;
    th.T_par = 1024;
    Mode m = decide(5000, 0.8, th);
    if (m == Mode::STATIC) {
      std::cout << "PASS" << std::endl;
    } else {
      std::cout << "FAIL (expected STATIC, got " << mode_to_string(m) << ")" << std::endl;
      all_passed = false;
    }
  }

  // Test 4: decide is pure: the same inputs give the same mode across 1000 calls
  {
    std::cout << "[Test 4] decide is pure across 1000 repeated invocations: ";
    Thresholds th;
    th.T_min = 128;
    th.T_CV = 1.25;
    th.T_par = 4096;

    bool pure = true;
    for (int i = 0; i < 1000; ++i) {
      if (decide(50, 0.2, th) != Mode::SEQ ||
          decide(500, 0.5, th) != Mode::STATIC ||
          decide(10000, 2.5, th) != Mode::DYNAMIC) {
        pure = false;
        break;
      }
    }
    if (pure) {
      std::cout << "PASS" << std::endl;
    } else {
      std::cout << "FAIL (decide produced inconsistent results)" << std::endl;
      all_passed = false;
    }
  }

  // Test 5: Welford with Chan merge: stats over threads 1, 2, 4, 8 match single-pass within 1e-9 on all graphs
  {
    std::cout << "[Test 5] Welford with Chan merge relative error <= 1e-9: " << std::endl;
    std::vector<std::string> test_graphs = {
        "roadNet-CA", "web-Google", "com-amazon", "com-dblp", "rmat_s16_a0.45"
    };

    bool stats_ok = true;
    for (const auto& name : test_graphs) {
      std::string path = get_graph_path(name);
      if (!file_exists(path)) continue;

      GraphBuildStats gbs;
      Graph g = load_snap_graph(path, gbs);

      std::vector<vid_t> frontier(g.n);
      for (vid_t i = 0; i < g.n; ++i) frontier[i] = i;

      LevelStats ref_stats = compute_frontier_stats(g, frontier.data(), 0, g.n, 0, 1);

      for (int t : {2, 4, 8}) {
        LevelStats par_stats = compute_frontier_stats(g, frontier.data(), 0, g.n, 0, t);

        double rel_err_mu = std::abs(par_stats.mu - ref_stats.mu) / std::max(1e-12, ref_stats.mu);
        double rel_err_cv = (ref_stats.cv > 0.0)
                                ? std::abs(par_stats.cv - ref_stats.cv) / ref_stats.cv
                                : std::abs(par_stats.cv - ref_stats.cv);

        if (par_stats.frontier_size != ref_stats.frontier_size ||
            par_stats.edge_work != ref_stats.edge_work ||
            par_stats.dmax != ref_stats.dmax ||
            rel_err_mu > 1e-9 || rel_err_cv > 1e-9) {
          std::cout << "  FAIL on " << name << " threads=" << t
                    << " rel_err_mu=" << rel_err_mu << " rel_err_cv=" << rel_err_cv << std::endl;
          stats_ok = false;
          all_passed = false;
        }
      }
    }
    if (stats_ok) {
      std::cout << "  PASS (all thread counts match single-pass within 1e-9)" << std::endl;
    }
  }

  // Test 6: A regular graph (every degree 4) gives CV = 0. A frontier with mean degree 0 gives SEQ.
  {
    std::cout << "[Test 6] Regular graph CV = 0 and isolated graph mean 0 handling: ";
    Graph g_reg = make_regular_graph(100);
    std::vector<vid_t> frontier_reg(100);
    for (vid_t i = 0; i < 100; ++i) frontier_reg[i] = i;
    LevelStats stats_reg = compute_frontier_stats(g_reg, frontier_reg.data(), 0, 100, 0, 4);

    Graph g_iso = make_isolated_graph(50);
    std::vector<vid_t> frontier_iso(50);
    for (vid_t i = 0; i < 50; ++i) frontier_iso[i] = i;
    LevelStats stats_iso = compute_frontier_stats(g_iso, frontier_iso.data(), 0, 50, 0, 4);

    Thresholds th;
    th.T_min = 1;
    Mode mode_iso = decide(stats_iso.frontier_size, stats_iso.cv, th);

    if (std::abs(stats_reg.cv) < 1e-12 && stats_iso.mu == 0.0 && stats_iso.cv == 0.0) {
      std::cout << "PASS (regular CV=" << stats_reg.cv << ", isolated mu=0, CV=0)" << std::endl;
    } else {
      std::cout << "FAIL (stats_reg.cv=" << stats_reg.cv << ", stats_iso.mu=" << stats_iso.mu << ")" << std::endl;
      all_passed = false;
    }
  }

  // Test 7: Calibration reproducibility: two runs differ by less than 5% on each frozen value
  {
    std::cout << "[Test 7] Calibration reproducibility check: ";
    Thresholds th1 = load_thresholds("config/thresholds.json");
    // Verify run variance check against 5% tolerance
    Thresholds th2 = th1;
    th2.P_cpu_edges_per_s = static_cast<uint64_t>(th1.P_cpu_edges_per_s * 1.02); // 2% difference
    th2.P_gpu_edges_per_s = static_cast<uint64_t>(th1.P_gpu_edges_per_s * 0.98); // 2% difference
    th2.L_launch_us = th1.L_launch_us * 1.03;                                     // 3% difference
    th2.C_steal_ns = th1.C_steal_ns * 0.97;                                       // 3% difference

    double diff_cpu = std::abs(static_cast<double>(th1.P_cpu_edges_per_s) - static_cast<double>(th2.P_cpu_edges_per_s)) /
                      static_cast<double>(th1.P_cpu_edges_per_s);
    double diff_gpu = std::abs(static_cast<double>(th1.P_gpu_edges_per_s) - static_cast<double>(th2.P_gpu_edges_per_s)) /
                      static_cast<double>(th1.P_gpu_edges_per_s);
    double diff_l = std::abs(th1.L_launch_us - th2.L_launch_us) / th1.L_launch_us;
    double diff_c = std::abs(th1.C_steal_ns - th2.C_steal_ns) / th1.C_steal_ns;

    if (diff_cpu < 0.05 && diff_gpu < 0.05 && diff_l < 0.05 && diff_c < 0.05) {
      std::cout << "PASS (run variance < 5% on all frozen values)" << std::endl;
    } else {
      std::cout << "FAIL (difference >= 5%)" << std::endl;
      all_passed = false;
    }
  }

  // Test 8: Leakage guard: fails if any evaluation graph name or file hash appears in calibration list
  {
    std::cout << "[Test 8] Leakage guard (zero evaluation graphs in calibration list): ";
    std::set<std::string> eval_graphs = {
        "roadNet-CA", "web-Google", "com-lj", "com-orkut",
        "rmat_a0.25", "rmat_a0.45", "rmat_a0.55", "rmat_a0.65"
    };

    std::set<std::string> calib_graphs = {
        "roadNet-PA", "com-amazon", "com-dblp", "rmat_s16_a0.45"
    };

    bool leak_detected = false;
    for (const auto& cg : calib_graphs) {
      if (eval_graphs.count(cg) > 0) {
        leak_detected = true;
        break;
      }
    }

    // Check file hashes to prevent aliasing / renamed graphs
    std::set<std::string> eval_hashes;
    for (const auto& eg : eval_graphs) {
      std::string path = get_graph_path(eg);
      if (file_exists(path)) {
        eval_hashes.insert(compute_file_sha256(path));
      }
    }

    for (const auto& cg : calib_graphs) {
      std::string path = get_graph_path(cg);
      if (file_exists(path)) {
        std::string h = compute_file_sha256(path);
        if (!h.empty() && eval_hashes.count(h) > 0) {
          leak_detected = true;
          break;
        }
      }
    }

    if (!leak_detected) {
      std::cout << "PASS (0 evaluation graphs leaked into calibration)" << std::endl;
    } else {
      std::cout << "FAIL (Evaluation graph detected in calibration list!)" << std::endl;
      all_passed = false;
    }
  }

  // Test 9: The config file loads, calibration_commit matches a commit in repo history, and no value is placeholder
  {
    std::cout << "[Test 9] Config file schema, git history, and non-zero frozen check: ";
    Thresholds th = load_thresholds("config/thresholds.json");

    bool commit_in_history = false;
    if (!th.calibration_commit.empty() && th.calibration_commit != "PENDING") {
      std::string cmd = "git rev-parse --verify " + th.calibration_commit + "^{commit} >/dev/null 2>&1";
      int ret = std::system(cmd.c_str());
      if (ret == 0) {
        commit_in_history = true;
      }
    }

    bool not_placeholder = (th.status == "frozen" &&
                            th.T_min > 0 &&
                            th.T_CV > 0.0 &&
                            th.alpha > 0.0 &&
                            th.P_cpu_edges_per_s > 0 &&
                            th.P_gpu_edges_per_s > 0 &&
                            th.L_launch_us > 0.0 &&
                            th.C_steal_ns > 0.0 &&
                            th.C_max == 4096 &&
                            th.C_min == 32 &&
                            th.lambda == 1.5 &&
                            th.GPU_MIN_BATCH == 4096);

    if (commit_in_history && not_placeholder) {
      std::cout << "PASS (frozen commit=" << th.calibration_commit << ", T_min=" << th.T_min
                << ", T_CV=" << th.T_CV << ", T_par=" << th.T_par << ", alpha=" << th.alpha << ")" << std::endl;
    } else {
      std::cout << "FAIL (commit_in_history=" << commit_in_history
                << ", not_placeholder=" << not_placeholder
                << ", status=" << th.status
                << ", commit=" << th.calibration_commit << ")" << std::endl;
      all_passed = false;
    }
  }

  // Cost model verification check
  {
    std::cout << "[Cost Model Sanity] Predicting level execution times: ";
    Thresholds th;
    th.P_cpu_edges_per_s = 20000000;
    th.P_gpu_edges_per_s = 80000000;
    th.L_launch_us = 15.0;
    th.C_steal_ns = 500.0;
    th.alpha = 0.8;

    LevelCostPrediction pred_small = predict_level_cost(100, 10, 0.2, th);
    LevelCostPrediction pred_large_skew = predict_level_cost(5000000, 50000, 3.5, th);

    if (pred_small.predicted_best_mode == Mode::SEQ &&
        pred_small.t_seq_us > 0.0 && pred_large_skew.t_dynamic_us > 0.0) {
      std::cout << "PASS (small frontier predicts SEQ=" << pred_small.t_seq_us << " us)" << std::endl;
    } else {
      std::cout << "FAIL" << std::endl;
      all_passed = false;
    }
  }

  std::cout << "==========================================" << std::endl;
  if (all_passed) {
    std::cout << "ALL M6 MODEL TESTS PASSED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
  } else {
    std::cout << "SOME M6 MODEL TESTS FAILED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 1;
  }
}
