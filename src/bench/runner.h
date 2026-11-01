#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "bench/manifest.h"
#include "graph/types.h"
#include "model/decide.h"
#include "sched/mode.h"

struct RunnerConfig {
  std::string manifest_path{"results/manifest.json"};
  std::string runs_jsonl_path{"results/runs.jsonl"};
  std::string csv_output_path{"results/levels.csv"};
  std::string env_inputs_path{"results/env/inputs.txt"};
  std::string thresholds_path{"config/thresholds.json"};
  bool allow_dirty{false};
  uint32_t timeout_sec{300};
  uint32_t warmups_per_policy{2};
  int threads{4};
  uint32_t gpu_min_batch{4096};
  bool dry_run{false};
  size_t expected_manifest_count{0};
  int kill_after_rows{0}; // Testing hook: self-terminate with SIGKILL after writing N rows
};

struct RunnerSummary {
  size_t total_scheduled{0};
  size_t runs_skipped_resumed{0};
  size_t runs_executed{0};
  size_t runs_completed{0};
  size_t runs_timed_out{0};
  size_t rows_written{0};
  size_t partial_rows_cleaned{0};
};

class BenchmarkRunner {
 public:
  explicit BenchmarkRunner(RunnerConfig config);
  ~BenchmarkRunner();

  // Initializes environment, validates git cleanliness, ensures inputs hashes, and cleans partial rows
  bool initialize(std::string* error_msg = nullptr);

  // Executes benchmark matrix according to manifest
  RunnerSummary run_all();

  // Executes a single run from the manifest
  bool execute_run(const BenchmarkRun& run, std::string* error_msg = nullptr);

  // Queries GPU VRAM via nvidia-smi to verify whether graph fits in VRAM
  static bool check_gpu_memory_fit(const std::string& graph_name, size_t* free_mb = nullptr, size_t* required_mb = nullptr);

 private:
  RunnerConfig config_;
  Thresholds thresholds_;
  std::vector<BenchmarkRun> manifest_;
  std::set<std::string> completed_run_ids_;
  std::map<std::pair<std::string, std::string>, int> warmup_tracker_; // (graph, policy) -> warmup count

  // Cached active graph to prevent redundant disk I/O across rounds
  std::string cached_graph_name_;
  Graph cached_graph_;
  std::vector<int32_t> cached_ref_dist_;
  std::map<std::string, std::vector<Mode>> oracle_modes_cache_; // graph -> per-level fastest mode

  void load_or_cache_graph(const std::string& graph_name);
  void compute_oracle_modes_for_graph(const std::string& graph_name);
  std::string resolve_graph_path(const std::string& graph_name);
};

// Returns true if git working tree has unstaged or staged modifications
bool is_git_tree_dirty();
