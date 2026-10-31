#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/bfs_state.h"
#include "cpu/bfs_omp.h"
#include "gpu/bfs_cuda.h"
#include "gpu/graph_gpu.h"
#include "graph/csr.h"
#include "graph/loader.h"
#include "graph/types.h"
#include "model/cost.h"
#include "model/decide.h"
#include "model/stats.h"
#include "sched/deque.h"
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

__global__ void empty_kernel() {}

namespace {

const std::vector<std::string> CALIBRATION_GRAPHS = {
    "roadNet-PA", "com-amazon", "com-dblp", "rmat_s16_a0.45"
};

const std::vector<std::string> EVALUATION_GRAPHS = {
    "roadNet-CA", "web-Google", "com-lj", "com-orkut",
    "rmat_a0.25", "rmat_a0.45", "rmat_a0.55", "rmat_a0.65"
};

const std::vector<size_t> T_MIN_GRID = {1, 16, 64, 256, 1024};
const std::vector<double> T_CV_GRID = {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0};
const std::vector<size_t> T_PAR_GRID = {0, 512, 2048, 8192, 32768};

std::string get_graph_file_path(const std::string& name) {
  if (name == "roadNet-PA") return "data/roadNet-PA.txt";
  if (name == "com-amazon") return "data/com-amazon.ungraph.txt";
  if (name == "com-dblp") return "data/com-dblp.ungraph.txt";
  if (name.rfind("rmat_", 0) == 0) return "data/" + name + ".txt";
  return "data/" + name + ".txt";
}

bool file_exists(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

void verify_leakage_guard() {
  std::set<std::string> eval_set(EVALUATION_GRAPHS.begin(), EVALUATION_GRAPHS.end());
  for (const auto& cg : CALIBRATION_GRAPHS) {
    if (eval_set.count(cg) > 0) {
      std::cerr << "FATAL: Evaluation graph detected in calibration list: " << cg << std::endl;
      std::exit(1);
    }
  }
}

// Microbenchmark 1: Kernel launch latency L in microseconds (median of 3 blocks of 20000)
double measure_launch_latency() {
  for (int i = 0; i < 2000; ++i) {
    empty_kernel<<<1, 1>>>();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  const int NUM_LAUNCHES = 20000;
  std::vector<double> block_us;
  for (int b = 0; b < 3; ++b) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_LAUNCHES; ++i) {
      empty_kernel<<<1, 1>>>();
      cudaDeviceSynchronize();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    block_us.push_back(total_us / static_cast<double>(NUM_LAUNCHES));
  }
  std::sort(block_us.begin(), block_us.end());
  return block_us[1]; // median
}

// Microbenchmark 2: Work-stealing cost C_steal in nanoseconds (median of 3 runs)
double measure_steal_cost() {
  const int TOTAL_STEALS = 200000;
  std::vector<double> runs;

  for (int b = 0; b < 3; ++b) {
    ChaseLevDeque deque(4096);
    std::atomic<int> steals_completed{0};

    for (uint32_t i = 0; i < 2000; ++i) {
      deque.push(Range{i, i + 1});
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    std::thread stealer([&]() {
      Range r;
      while (steals_completed.load(std::memory_order_relaxed) < TOTAL_STEALS) {
        if (deque.steal(r)) {
          steals_completed.fetch_add(1, std::memory_order_relaxed);
        } else {
          std::this_thread::yield();
        }
      }
    });

    std::thread producer([&]() {
      uint32_t cur = 2000;
      while (steals_completed.load(std::memory_order_relaxed) < TOTAL_STEALS) {
        if (deque.size() < 1000 && cur < TOTAL_STEALS + 5000) {
          for (int k = 0; k < 1000; ++k) {
            deque.push(Range{cur, cur + 1});
            cur++;
          }
        } else {
          std::this_thread::yield();
        }
      }
    });

    producer.join();
    stealer.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    runs.push_back(total_ns / static_cast<double>(TOTAL_STEALS));
  }
  std::sort(runs.begin(), runs.end());
  return runs[1]; // median
}

struct LevelData {
  size_t frontier_size;
  double cv;
  double t_seq_us;
  double t_static_us;
  double t_dynamic_us;
};

}  // namespace

int main(int argc, char** argv) {
  verify_leakage_guard();

  std::string output_json = "";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--output=", 0) == 0) {
      output_json = arg.substr(9);
    }
  }

  // 1. Measure L (launch latency)
  double L_launch_us = measure_launch_latency();

  // 2. Measure C_steal (steal cost)
  double C_steal_ns = measure_steal_cost();

  // 3. Measure P_cpu, P_gpu, stats cost, and forced-mode level times on calibration graphs
  uint64_t total_cpu_edges = 0;
  double total_cpu_sec = 0.0;

  uint64_t total_gpu_edges = 0;
  double total_gpu_sec = 0.0;

  double total_stats_us = 0.0;
  size_t total_stats_count = 0;

  std::vector<LevelData> levels;

  Params p;
  p.threads = 4;
  p.alpha = 0.50;

  for (const auto& name : CALIBRATION_GRAPHS) {
    std::string path = get_graph_file_path(name);
    if (!file_exists(path)) {
      std::cerr << "Calibration graph missing: " << path << std::endl;
      continue;
    }

    GraphBuildStats gbs;
    Graph g = load_snap_graph(path, gbs);

    // Measure stats pass overhead
    {
      std::vector<vid_t> f(g.n);
      for (vid_t i = 0; i < g.n; ++i) f[i] = i;
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int rep = 0; rep < 5; ++rep) {
        LevelStats s = compute_frontier_stats(g, f.data(), 0, g.n, 0, p.threads);
        (void)s;
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 5.0;
      total_stats_us += us;
      total_stats_count++;
    }

    // Run 3 repetitions and use median per level
    const int REPS = 3;
    std::vector<std::vector<LevelLog>> rep_logs_seq(REPS);
    std::vector<std::vector<LevelLog>> rep_logs_static(REPS);
    std::vector<std::vector<LevelLog>> rep_logs_dynamic(REPS);

    Params p_seq = p;
    p_seq.threads = 1;

    for (int r = 0; r < REPS; ++r) {
      bfs_run(g, 0, Mode::SEQ, p_seq, &rep_logs_seq[r]);
      bfs_run(g, 0, Mode::STATIC, p, &rep_logs_static[r]);
      bfs_run(g, 0, Mode::DYNAMIC, p, &rep_logs_dynamic[r]);
    }

    // Accumulate CPU edge throughput from SEQ
    for (int r = 0; r < REPS; ++r) {
      for (const auto& l : rep_logs_seq[r]) {
        if (l.frontier_size > 0 && l.t_cpu_us > 0.0) {
          total_cpu_edges += l.edge_work;
          total_cpu_sec += (l.t_cpu_us * 1e-6);
        }
      }
    }

    // Accumulate GPU edge throughput from STATIC
    for (int r = 0; r < REPS; ++r) {
      for (const auto& l : rep_logs_static[r]) {
        if (l.frontier_size >= 4096 && l.t_gpu_us > 0.0) {
          uint64_t gpu_edges = static_cast<uint64_t>(l.edge_work * p.alpha);
          total_gpu_edges += gpu_edges;
          total_gpu_sec += (l.t_gpu_us * 1e-6);
        }
      }
    }

    // Median per-level times across repetitions
    size_t num_levels = std::min({rep_logs_seq[0].size(), rep_logs_static[0].size(), rep_logs_dynamic[0].size()});
    for (size_t i = 0; i < num_levels; ++i) {
      if (rep_logs_seq[0][i].frontier_size > 0) {
        LevelData ld;
        ld.frontier_size = rep_logs_seq[0][i].frontier_size;
        ld.cv = rep_logs_seq[0][i].cv;

        std::vector<double> v_seq(REPS), v_static(REPS), v_dyn(REPS);
        for (int r = 0; r < REPS; ++r) {
          v_seq[r] = rep_logs_seq[r][i].time_us;
          v_static[r] = rep_logs_static[r][i].time_us;
          v_dyn[r] = rep_logs_dynamic[r][i].time_us;
        }
        std::sort(v_seq.begin(), v_seq.end());
        std::sort(v_static.begin(), v_static.end());
        std::sort(v_dyn.begin(), v_dyn.end());

        ld.t_seq_us = v_seq[1];
        ld.t_static_us = v_static[1];
        ld.t_dynamic_us = v_dyn[1];
        levels.push_back(ld);
      }
    }
  }

  uint64_t P_cpu = (total_cpu_sec > 0.0) ? static_cast<uint64_t>(total_cpu_edges / total_cpu_sec) : 25000000;
  uint64_t P_gpu = (total_gpu_sec > 0.0) ? static_cast<uint64_t>(total_gpu_edges / total_gpu_sec) : 5000000;
  double alpha = static_cast<double>(P_gpu) / static_cast<double>(P_cpu + P_gpu);
  double avg_stats_cost_us = (total_stats_count > 0) ? (total_stats_us / total_stats_count) : 0.0;

  // 4. Threshold Grid Search minimizing oracle regret over calibration levels
  double min_regret = std::numeric_limits<double>::infinity();
  size_t best_T_min = 64;
  double best_T_CV = 0.75;
  size_t best_T_par = 2048;

  for (size_t t_min : T_MIN_GRID) {
    for (double t_cv : T_CV_GRID) {
      for (size_t t_par : T_PAR_GRID) {
        Thresholds cand;
        cand.T_min = t_min;
        cand.T_CV = t_cv;
        cand.T_par = t_par;

        double regret = 0.0;
        for (const auto& l : levels) {
          Mode chosen = decide(l.frontier_size, l.cv, cand);
          double chosen_t = (chosen == Mode::SEQ) ? l.t_seq_us : ((chosen == Mode::STATIC) ? l.t_static_us : l.t_dynamic_us);
          double best_t = std::min({l.t_seq_us, l.t_static_us, l.t_dynamic_us});
          regret += (chosen_t - best_t);
        }

        if (regret < min_regret) {
          min_regret = regret;
          best_T_min = t_min;
          best_T_CV = t_cv;
          best_T_par = t_par;
        }
      }
    }
  }

  // Format and output JSON
  std::stringstream ss;
  ss << "{\n";
  ss << "  \"T_min\": " << best_T_min << ",\n";
  ss << "  \"T_CV\": " << best_T_CV << ",\n";
  ss << "  \"T_par\": " << best_T_par << ",\n";
  ss << "  \"alpha\": " << std::round(alpha * 10000.0) / 10000.0 << ",\n";
  ss << "  \"P_cpu_edges_per_s\": " << P_cpu << ",\n";
  ss << "  \"P_gpu_edges_per_s\": " << P_gpu << ",\n";
  ss << "  \"L_launch_us\": " << std::round(L_launch_us * 100.0) / 100.0 << ",\n";
  ss << "  \"C_steal_ns\": " << std::round(C_steal_ns * 10.0) / 10.0 << ",\n";
  ss << "  \"C_max\": 4096,\n";
  ss << "  \"C_min\": 32,\n";
  ss << "  \"lambda\": 1.5,\n";
  ss << "  \"GPU_MIN_BATCH\": 4096,\n";
  ss << "  \"stats_cost_us_avg\": " << std::round(avg_stats_cost_us * 100.0) / 100.0 << ",\n";
  ss << "  \"min_oracle_regret_us\": " << min_regret << "\n";
  ss << "}\n";

  std::cout << ss.str();

  if (!output_json.empty()) {
    std::ofstream out(output_json);
    out << ss.str();
  }

  return 0;
}
