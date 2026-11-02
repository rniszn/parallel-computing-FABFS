#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/bfs_state.h"
#include "cpu/bfs_omp.h"
#include "gpu/bfs_cuda.h"
#include "gpu/graph_gpu.h"
#include "graph/csr.h"
#include "graph/loader.h"
#include "graph/rmat.h"
#include "graph/types.h"
#include "log/record.h"
#include "model/decide.h"
#include "ref/bfs_ref.h"
#include "sched/mode.h"
#include "sched/scheduler.h"

#define CHECK_CUDA(call)                                                         \
  do {                                                                           \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));                                     \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

struct CompareConfig {
  std::string graph_name;
  std::string graph_path;
  int rounds{15};
  int warmup{2};
  uint64_t seed{12345};
  std::string thresholds_path{"config/thresholds.json"};
  std::string out_dir{"results/compare"};
  std::string csv_path;
  std::string json_path;
  int threads{4};
  bool quick{false};
};

static void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " [options]\n"
            << "Options:\n"
            << "  --graph=<name>         Graph name (roadNet-CA, web-Google, rmat_a0.55)\n"
            << "  --graph-path=<path>    Path to graph edge-list file\n"
            << "  --rounds=<N>           Number of timed rounds (default: 15, --quick sets 3)\n"
            << "  --warmup=<W>           Number of unrecorded warm-up rounds (default: 2)\n"
            << "  --seed=<S>             RNG seed for policy shuffling (default: 12345)\n"
            << "  --thresholds=<path>    Path to config/thresholds.json (default: config/thresholds.json)\n"
            << "  --out-dir=<dir>        Output directory for CSV and JSON results\n"
            << "  --threads=<T>          CPU worker threads (default: 4)\n"
            << "  --quick                Run quick sanity evaluation (warmup=2, rounds=3)\n"
            << "  --help                 Show this help message\n";
}

static bool verify_correctness(const std::vector<int32_t>& dist,
                              const std::vector<int32_t>& ref) {
  if (dist.size() != ref.size()) return false;
  for (size_t i = 0; i < dist.size(); ++i) {
    if (dist[i] != ref[i]) return false;
  }
  return true;
}

int main(int argc, char** argv) {
  CompareConfig cfg;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--quick") {
      cfg.quick = true;
      cfg.rounds = 3;
      cfg.warmup = 2;
    } else if (arg.rfind("--graph=", 0) == 0) {
      cfg.graph_name = arg.substr(8);
    } else if (arg.rfind("--graph-path=", 0) == 0) {
      cfg.graph_path = arg.substr(13);
    } else if (arg.rfind("--rounds=", 0) == 0) {
      cfg.rounds = std::stoi(arg.substr(9));
    } else if (arg.rfind("--warmup=", 0) == 0) {
      cfg.warmup = std::stoi(arg.substr(9));
    } else if (arg.rfind("--seed=", 0) == 0) {
      cfg.seed = std::stoull(arg.substr(7));
    } else if (arg.rfind("--thresholds=", 0) == 0) {
      cfg.thresholds_path = arg.substr(13);
    } else if (arg.rfind("--out-dir=", 0) == 0) {
      cfg.out_dir = arg.substr(10);
    } else if (arg.rfind("--threads=", 0) == 0) {
      cfg.threads = std::stoi(arg.substr(10));
    }
  }

  if (cfg.graph_name.empty()) {
    std::cerr << "Error: --graph=<name> is required.\n";
    print_usage(argv[0]);
    return 1;
  }

  if (cfg.graph_path.empty()) {
    if (cfg.graph_name == "roadNet-CA") cfg.graph_path = "data/roadNet-CA.txt";
    else if (cfg.graph_name == "web-Google") cfg.graph_path = "data/web-Google.txt";
    else if (cfg.graph_name == "rmat_a0.55" || cfg.graph_name == "rmat20_a055") {
      cfg.graph_path = "data/rmat_a0.55.txt";
    } else {
      cfg.graph_path = "data/" + cfg.graph_name + ".txt";
    }
  }

  if (cfg.csv_path.empty()) {
    cfg.csv_path = cfg.out_dir + "/levels.csv";
  }
  if (cfg.json_path.empty()) {
    cfg.json_path = cfg.out_dir + "/raw_runs.json";
  }

  // 1. Device and M0 Managed Memory Capability Check
  int dev_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&dev_count));
  if (dev_count == 0) {
    std::cerr << "\033[1;31m[ERROR]\033[0m No CUDA-capable device detected.\n";
    return 1;
  }

  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

  int managed = 0, concurrent = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, dev));
  CHECK_CUDA(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, dev));

  if (!managed) {
    std::cerr << "\033[1;31m[ERROR]\033[0m Device " << prop.name
              << " does not support CUDA Unified Memory (managedMemory == 0).\n";
    return 1;
  }

  // 2. Check GPU VRAM Availability
  size_t free_bytes = 0, total_bytes = 0;
  CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
  size_t free_mb = free_bytes / (1024 * 1024);
  size_t total_mb = total_bytes / (1024 * 1024);

  // 3. Load or Generate Graph (Timing excluded from BFS traversal metrics)
  std::cout << "[compare_runner] Loading graph '" << cfg.graph_name
            << "' from " << cfg.graph_path << "...\n";
  Graph g;
  GraphBuildStats gbs;

  if (cfg.graph_name == "rmat_a0.55" || cfg.graph_name == "rmat20_a055" ||
      cfg.graph_path.find("rmat_a0.55") != std::string::npos) {
    g = generate_or_load_rmat(cfg.graph_path, 20, 16, 0.55, 12345, gbs);
  } else {
    g = load_snap_graph(cfg.graph_path, gbs);
  }

  std::cout << "[compare_runner] Graph loaded: |V|=" << g.n << ", |E|=" << g.m << "\n";

  // Check required VRAM for graph CSR and BFS state
  size_t required_bytes = (g.n + 1) * sizeof(eid_t) + g.m * sizeof(vid_t) + g.n * 16;
  size_t required_mb = required_bytes / (1024 * 1024);
  if (free_mb < required_mb + 256) {
    std::cerr << "\033[1;31m[ERROR: INSUFFICIENT VRAM]\033[0m Graph " << cfg.graph_name
              << " requires ~" << required_mb << " MB VRAM, but GPU only has "
              << free_mb << " MB free out of " << total_mb << " MB.\n";
    return 1;
  }

  // 4. Ground-Truth Reference BFS (M2)
  std::cout << "[compare_runner] Computing M2 serial reference BFS from root vertex 0...\n";
  std::vector<int32_t> ref_dist = bfs_ref(g, 0);

  // 5. Pre-upload Graph to GPU (M4) so traversal timing excludes upload
  std::cout << "[compare_runner] Uploading graph to GPU unified memory...\n";
  GpuGraph gg = upload_graph(g);

  // 6. Load Frozen Calibration Thresholds (M6)
  std::cout << "[compare_runner] Loading frozen calibration thresholds from "
            << cfg.thresholds_path << "...\n";
  Thresholds th = load_thresholds(cfg.thresholds_path);

  Params p;
  p.threads = cfg.threads;
  p.alpha = th.alpha;
  p.lambda = th.lambda;
  p.C_max = th.C_max;
  p.C_min = th.C_min;
  p.GPU_MIN_BATCH = th.GPU_MIN_BATCH;

  // 7. Ensure output directory and open LevelLogger (M7)
  std::string mkdir_cmd = "mkdir -p " + cfg.out_dir;
  system(mkdir_cmd.c_str());

  LevelLogger logger;
  logger.open(cfg.csv_path);

  std::string git_commit = get_build_git_commit();
  std::string driver_ver = get_driver_version();

  const std::vector<std::string> policies = {"SEQ", "STATIC", "DYNAMIC", "FABFS"};

  // 8. Unrecorded Warmup Runs (2 rounds)
  std::cout << "[compare_runner] Performing " << cfg.warmup << " unrecorded warm-up rounds...\n";
  for (int w = 0; w < cfg.warmup; ++w) {
    for (const auto& pol : policies) {
      std::vector<int32_t> dist;
      if (pol == "SEQ") dist = bfs_run(g, 0, Mode::SEQ, p, nullptr);
      else if (pol == "STATIC") dist = bfs_run(g, 0, Mode::STATIC, p, nullptr);
      else if (pol == "DYNAMIC") dist = bfs_run(g, 0, Mode::DYNAMIC, p, nullptr);
      else if (pol == "FABFS") dist = bfs_adaptive(g, 0, p, th, nullptr);

      if (!verify_correctness(dist, ref_dist)) {
        std::cerr << "\033[1;31m[CORRECTNESS FAIL]\033[0m Warmup mismatch for policy "
                  << pol << "! Aborting.\n";
        return 2;
      }
    }
  }

  // 9. Timed Evaluation Rounds with Seeded Random Order
  std::cout << "[compare_runner] Executing " << cfg.rounds
            << " timed evaluation rounds (seeded random order)...\n";

  // Data structures to collect round times and per-level stats
  std::map<std::string, std::vector<double>> policy_total_times_ms;
  for (const auto& pol : policies) policy_total_times_ms[pol] = {};

  // Level telemetry accumulator: [policy][level] -> list of LevelLog
  std::map<std::string, std::map<uint32_t, std::vector<LevelLog>>> level_logs_by_policy;

  for (int r = 0; r < cfg.rounds; ++r) {
    std::vector<std::string> round_policies = policies;
    std::mt19937 rng(cfg.seed + r);
    std::shuffle(round_policies.begin(), round_policies.end(), rng);

    for (const auto& pol : round_policies) {
      std::vector<LevelLog> logs;
      std::vector<int32_t> dist;

      auto t_start = std::chrono::high_resolution_clock::now();
      if (pol == "SEQ") {
        dist = bfs_run(g, 0, Mode::SEQ, p, &logs);
      } else if (pol == "STATIC") {
        dist = bfs_run(g, 0, Mode::STATIC, p, &logs);
      } else if (pol == "DYNAMIC") {
        dist = bfs_run(g, 0, Mode::DYNAMIC, p, &logs);
      } else if (pol == "FABFS") {
        dist = bfs_adaptive(g, 0, p, th, &logs);
      }
      auto t_end = std::chrono::high_resolution_clock::now();

      double wall_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
      policy_total_times_ms[pol].push_back(wall_ms);

      // Verify M2 Correctness
      if (!verify_correctness(dist, ref_dist)) {
        std::cerr << "\033[1;31m[CORRECTNESS FAIL]\033[0m Run mismatch for policy "
                  << pol << " in round " << r << "! Aborting.\n";
        return 2;
      }

      // Log per-level rows to M7 CSV
      for (const auto& l : logs) {
        LevelRow row;
        row.run_id = "comp_" + cfg.graph_name + "_" + pol + "_r" + std::to_string(r);
        row.graph = cfg.graph_name;
        row.policy = (pol == "FABFS") ? "Adaptive" : pol;
        row.round = static_cast<uint32_t>(r);
        row.level = l.level;
        row.frontier_size = l.frontier_size;
        row.edge_work = l.edge_work;
        row.mu = l.mu;
        row.cv = l.cv;
        row.dmax = l.dmax;
        row.mode_chosen = (l.mode_chosen == Mode::SEQ) ? "SEQ" :
                          (l.mode_chosen == Mode::STATIC ? "STATIC" : "DYNAMIC");
        row.chunk_size = l.chunk_size;
        row.time_us = l.time_us;
        row.steals_attempted = l.steals_attempted;
        row.steals_succeeded = l.steals_succeeded;
        row.gpu_batches = l.gpu_batches;
        row.correct = true;
        row.git_commit = git_commit;
        row.driver_version = driver_ver;
        row.threads = cfg.threads;
        row.alpha = th.alpha;
        row.gpu_min_batch = th.GPU_MIN_BATCH;
        row.num_chunks = l.num_chunks;
        row.gpu_batch_vertices = l.gpu_batch_vertices;
        row.t_cpu_us = l.t_cpu_us;
        row.t_gpu_us = l.t_gpu_us;
        row.t_merge_us = 0.0;
        row.gpu_idle_us = l.gpu_idle_us;

        logger.write_row(row);

        level_logs_by_policy[pol][l.level].push_back(l);
      }
    }
  }

  logger.close();
  std::cout << "[compare_runner] Wrote per-level M7 CSV rows to " << cfg.csv_path << "\n";

  // 10. Compute Oracle Per-Level Times and Derive ORACLE Round Totals
  uint32_t total_levels = 0;
  for (const auto& kv : level_logs_by_policy["FABFS"]) {
    if (kv.first + 1 > total_levels) total_levels = kv.first + 1;
  }

  std::vector<double> oracle_times_ms(cfg.rounds, 0.0);
  for (int r = 0; r < cfg.rounds; ++r) {
    double r_oracle_us = 0.0;
    for (uint32_t l = 0; l < total_levels; ++l) {
      double t_seq = (r < static_cast<int>(level_logs_by_policy["SEQ"][l].size()))
                         ? level_logs_by_policy["SEQ"][l][r].time_us : 1e12;
      double t_static = (r < static_cast<int>(level_logs_by_policy["STATIC"][l].size()))
                            ? level_logs_by_policy["STATIC"][l][r].time_us : 1e12;
      double t_dyn = (r < static_cast<int>(level_logs_by_policy["DYNAMIC"][l].size()))
                         ? level_logs_by_policy["DYNAMIC"][l][r].time_us : 1e12;
      r_oracle_us += std::min({t_seq, t_static, t_dyn});
    }
    oracle_times_ms[r] = r_oracle_us / 1000.0;
  }
  policy_total_times_ms["ORACLE"] = oracle_times_ms;

  // 11. Write Raw Metrics to JSON for Python Formatter
  std::ofstream jout(cfg.json_path);
  if (!jout.is_open()) {
    std::cerr << "Failed to open " << cfg.json_path << " for writing\n";
    return 1;
  }

  jout << "{\n";
  jout << "  \"graph\": \"" << cfg.graph_name << "\",\n";
  jout << "  \"vertices\": " << g.n << ",\n";
  jout << "  \"edges\": " << g.m << ",\n";
  jout << "  \"total_levels\": " << total_levels << ",\n";
  jout << "  \"rounds\": " << cfg.rounds << ",\n";
  jout << "  \"warmup\": " << cfg.warmup << ",\n";
  jout << "  \"seed\": " << cfg.seed << ",\n";
  jout << "  \"gpu_name\": \"" << prop.name << "\",\n";
  jout << "  \"driver_version\": \"" << driver_ver << "\",\n";
  jout << "  \"git_commit\": \"" << git_commit << "\",\n";
  jout << "  \"managed_memory\": " << managed << ",\n";
  jout << "  \"concurrent_managed_access\": " << concurrent << ",\n";

  // Policy total times
  jout << "  \"policy_times_ms\": {\n";
  bool first_pol = true;
  for (const auto& kv : policy_total_times_ms) {
    if (!first_pol) jout << ",\n";
    first_pol = false;
    jout << "    \"" << kv.first << "\": [";
    for (size_t i = 0; i < kv.second.size(); ++i) {
      if (i > 0) jout << ", ";
      jout << std::fixed << std::setprecision(3) << kv.second[i];
    }
    jout << "]";
  }
  jout << "\n  },\n";

  // Per-level aggregated summary
  jout << "  \"per_level\": [\n";
  for (uint32_t l = 0; l < total_levels; ++l) {
    auto get_median_time = [&](const std::string& pol) -> double {
      auto& logs = level_logs_by_policy[pol][l];
      if (logs.empty()) return 0.0;
      std::vector<double> times;
      for (const auto& row : logs) times.push_back(row.time_us);
      std::sort(times.begin(), times.end());
      return times[times.size() / 2];
    };

    double t_seq = get_median_time("SEQ");
    double t_static = get_median_time("STATIC");
    double t_dyn = get_median_time("DYNAMIC");
    double t_fabfs = get_median_time("FABFS");

    std::string best_mode = "SEQ";
    double best_time = t_seq;
    if (t_static < best_time) {
      best_time = t_static;
      best_mode = "STATIC";
    }
    if (t_dyn < best_time) {
      best_time = t_dyn;
      best_mode = "DYNAMIC";
    }

    // Reference frontier topology stats from FABFS (or SEQ)
    const auto& fabfs_log = level_logs_by_policy["FABFS"][l].front();
    std::string fabfs_mode = (fabfs_log.mode_chosen == Mode::SEQ) ? "SEQ" :
                             (fabfs_log.mode_chosen == Mode::STATIC ? "STATIC" : "DYNAMIC");

    // Steals and GPU batch aggregates
    uint64_t dyn_steals_att = 0, dyn_steals_succ = 0;
    for (const auto& r_log : level_logs_by_policy["DYNAMIC"][l]) {
      dyn_steals_att += r_log.steals_attempted;
      dyn_steals_succ += r_log.steals_succeeded;
    }
    uint64_t fab_steals_att = 0, fab_steals_succ = 0;
    uint32_t fab_gpu_batches = 0;
    double fab_gpu_idle_us = 0.0;
    for (const auto& r_log : level_logs_by_policy["FABFS"][l]) {
      fab_steals_att += r_log.steals_attempted;
      fab_steals_succ += r_log.steals_succeeded;
      fab_gpu_batches += r_log.gpu_batches;
      fab_gpu_idle_us += r_log.gpu_idle_us;
    }

    double regret_us = std::max(0.0, t_fabfs - best_time);

    jout << "    {\n";
    jout << "      \"level\": " << l << ",\n";
    jout << "      \"frontier_size\": " << fabfs_log.frontier_size << ",\n";
    jout << "      \"edge_work\": " << fabfs_log.edge_work << ",\n";
    jout << "      \"mu\": " << std::fixed << std::setprecision(2) << fabfs_log.mu << ",\n";
    jout << "      \"cv\": " << std::fixed << std::setprecision(2) << fabfs_log.cv << ",\n";
    jout << "      \"dmax\": " << fabfs_log.dmax << ",\n";
    jout << "      \"t_seq_us\": " << std::fixed << std::setprecision(1) << t_seq << ",\n";
    jout << "      \"t_static_us\": " << std::fixed << std::setprecision(1) << t_static << ",\n";
    jout << "      \"t_dynamic_us\": " << std::fixed << std::setprecision(1) << t_dyn << ",\n";
    jout << "      \"t_fabfs_us\": " << std::fixed << std::setprecision(1) << t_fabfs << ",\n";
    jout << "      \"best_forced_mode\": \"" << best_mode << "\",\n";
    jout << "      \"fabfs_mode\": \"" << fabfs_mode << "\",\n";
    jout << "      \"match\": " << ((fabfs_mode == best_mode) ? "true" : "false") << ",\n";
    jout << "      \"regret_us\": " << std::fixed << std::setprecision(1) << regret_us << ",\n";
    jout << "      \"dyn_steals_att\": " << dyn_steals_att / cfg.rounds << ",\n";
    jout << "      \"dyn_steals_succ\": " << dyn_steals_succ / cfg.rounds << ",\n";
    jout << "      \"fab_steals_att\": " << fab_steals_att / cfg.rounds << ",\n";
    jout << "      \"fab_steals_succ\": " << fab_steals_succ / cfg.rounds << ",\n";
    jout << "      \"fab_gpu_batches\": " << fab_gpu_batches / cfg.rounds << ",\n";
    jout << "      \"fab_gpu_idle_us\": " << std::fixed << std::setprecision(1) << (fab_gpu_idle_us / cfg.rounds) << "\n";
    jout << "    }" << ((l + 1 < total_levels) ? "," : "") << "\n";
  }
  jout << "  ]\n";
  jout << "}\n";
  jout.close();

  std::cout << "[compare_runner] Wrote raw metrics to " << cfg.json_path << "\n";
  return 0;
}
