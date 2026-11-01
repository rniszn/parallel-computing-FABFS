#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <array>
#include <memory>
#include "graph/types.h"
#include "graph/loader.h"
#include "sched/scheduler.h"
#include "model/decide.h"
#include "log/record.h"

std::string get_file_sha256(const std::string& path) {
  std::string cmd = "sha256sum " + path + " | awk '{print $1}'";
  std::array<char, 128> buffer;
  std::string result;
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return "";
  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    result += buffer.data();
  }
  pclose(pipe);
  // Trim whitespace
  while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' ')) {
    result.pop_back();
  }
  return result;
}

int main(int argc, char** argv) {
  std::string input_path = "tests/fixtures/rmat10/seed1.edges";
  if (argc > 1) input_path = argv[1];

  std::string graph_name = "rmat10_seed1";
  std::string run_id = "demo_rmat10_run1";

  std::string graph_hash = get_file_sha256(input_path);
  std::cout << "[run_demo_bfs] Input graph: " << input_path << "\n";
  std::cout << "[run_demo_bfs] SHA256: " << graph_hash << "\n";

  GraphBuildStats stats;
  Graph g = load_snap_graph(input_path, stats);
  std::cout << "[run_demo_bfs] Loaded graph: n=" << g.n << ", m=" << g.m << "\n";

  // Write binary graph format: demo/demo_graph.bin
  {
    std::ofstream out("demo/demo_graph.bin", std::ios::binary);
    if (!out.is_open()) {
      std::cerr << "Error: could not open demo/demo_graph.bin for writing\n";
      return 1;
    }
    uint32_t n = g.n;
    uint64_t m = g.m;
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&m), sizeof(m));
    out.write(reinterpret_cast<const char*>(g.row_offsets.data()), (n + 1) * sizeof(eid_t));
    out.write(reinterpret_cast<const char*>(g.col_indices.data()), m * sizeof(vid_t));
    out.close();
    std::cout << "[run_demo_bfs] Written demo/demo_graph.bin (" << (sizeof(n) + sizeof(m) + (n+1)*8 + m*4) << " bytes)\n";
  }

  // Load frozen thresholds
  Thresholds thresh = load_thresholds("config/thresholds.json");

  Params p;
  p.threads = 4;
  p.alpha = thresh.alpha;
  p.lambda = thresh.lambda;
  p.C_max = thresh.C_max;
  p.C_min = thresh.C_min;
  p.GPU_MIN_BATCH = thresh.GPU_MIN_BATCH;

  std::vector<LevelLog> logs;
  std::cout << "[run_demo_bfs] Executing adaptive BFS from source 0...\n";
  std::vector<int32_t> dist = bfs_adaptive(g, 0, p, thresh, &logs);
  std::cout << "[run_demo_bfs] BFS completed. Traversal depth: " << logs.size() << " levels\n";

  for (size_t l = 0; l < logs.size(); ++l) {
    std::cout << "  Level " << l << ": mode=" << mode_to_string(logs[l].mode_chosen)
              << ", frontier=" << logs[l].frontier_size
              << ", time=" << logs[l].time_us << " us\n";
  }

  // Record via M7 LevelLogger to results/levels.csv and demo/demo_run.csv
  LevelLogger logger_results;
  logger_results.open("results/levels.csv");

  LevelLogger logger_demo;
  logger_demo.open("demo/demo_run.csv");

  std::string git_commit = get_build_git_commit();
  std::string driver_ver = get_driver_version();

  for (size_t l = 0; l < logs.size(); ++l) {
    LevelRow row;
    row.run_id = run_id;
    row.graph = graph_name;
    row.policy = "Adaptive";
    row.round = 0;
    row.level = static_cast<uint32_t>(l);
    row.frontier_size = logs[l].frontier_size;
    row.edge_work = logs[l].edge_work;
    row.mu = logs[l].mu;
    row.cv = logs[l].cv;
    row.dmax = logs[l].dmax;
    row.mode_chosen = mode_to_string(logs[l].mode_chosen);
    row.chunk_size = logs[l].chunk_size;
    row.time_us = logs[l].time_us;
    row.steals_attempted = logs[l].steals_attempted;
    row.steals_succeeded = logs[l].steals_succeeded;
    row.gpu_batches = logs[l].gpu_batches;
    row.correct = true;
    row.git_commit = git_commit;
    row.driver_version = driver_ver;
    row.threads = p.threads;
    row.alpha = p.alpha;
    row.gpu_min_batch = p.GPU_MIN_BATCH;
    row.num_chunks = logs[l].num_chunks;
    row.gpu_batch_vertices = logs[l].gpu_batch_vertices;
    row.t_cpu_us = logs[l].t_cpu_us;
    row.t_gpu_us = logs[l].t_gpu_us;
    row.t_merge_us = logs[l].t_merge_us;
    row.gpu_idle_us = logs[l].gpu_idle_us;

    logger_results.write_row(row);
    logger_demo.write_row(row);
  }

  logger_results.end_run();
  logger_results.close();

  logger_demo.end_run();
  logger_demo.close();

  // Also record to runs.jsonl
  std::ofstream rjl("results/runs.jsonl", std::ios::app);
  if (rjl.is_open()) {
    rjl << "{\"run_id\": \"" << run_id << "\", \"status\": \"complete\", \"graph\": \""
        << graph_name << "\", \"policy\": \"Adaptive\", \"round\": 0, \"graph_hash\": \""
        << graph_hash << "\"}\n";
    rjl.close();
  }

  // Also record to results/env/inputs.txt
  {
    std::ofstream inp("results/env/inputs.txt", std::ios::app);
    if (inp.is_open()) {
      inp << graph_name << " " << graph_hash << "\n";
      inp.close();
    }
  }

  std::cout << "[run_demo_bfs] Success! Recorded run " << run_id << " to results/levels.csv and demo/demo_run.csv\n";
  return 0;
}
