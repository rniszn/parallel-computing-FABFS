#include "runner.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

#include "common/bfs_state.h"
#include "gpu/graph_gpu.h"
#include "graph/loader.h"
#include "log/record.h"
#include "ref/bfs_ref.h"
#include "resume.h"
#include "sched/scheduler.h"

bool is_git_tree_dirty() {
  FILE* fp = popen("git status --porcelain 2>/dev/null", "r");
  if (!fp) return false;

  char buf[256];
  bool dirty = false;
  while (fgets(buf, sizeof(buf), fp)) {
    // Untracked files (??) don't count as dirty if ignored or clean, but modified/staged files do
    if (buf[0] != '?' && buf[0] != '\0' && buf[0] != '\n') {
      dirty = true;
      break;
    }
  }
  pclose(fp);
  return dirty;
}

BenchmarkRunner::BenchmarkRunner(RunnerConfig config) : config_(std::move(config)) {}

BenchmarkRunner::~BenchmarkRunner() = default;

bool BenchmarkRunner::check_gpu_memory_fit(const std::string& graph_name, size_t* free_mb, size_t* required_mb) {
  size_t free_mem = 0;
  size_t total_mem = 0;

  FILE* fp = popen("nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null", "r");
  if (fp) {
    char buf[128];
    if (fgets(buf, sizeof(buf), fp)) {
      std::sscanf(buf, "%zu, %zu", &free_mem, &total_mem);
    }
    pclose(fp);
  }

  // Estimated memory requirements per graph
  size_t est_req = 512; // default 512 MB
  if (graph_name.find("orkut") != std::string::npos) {
    est_req = 1200; // com-orkut ~1.2 GB
  } else if (graph_name.find("lj") != std::string::npos || graph_name.find("LiveJournal") != std::string::npos) {
    est_req = 600;  // com-lj ~600 MB
  }

  if (free_mb) *free_mb = free_mem;
  if (required_mb) *required_mb = est_req;

  if (free_mem > 0 && free_mem < est_req) {
    return false;
  }
  return true;
}

std::string BenchmarkRunner::resolve_graph_path(const std::string& graph_name) {
  if (graph_name == "roadNet-CA") return "data/roadNet-CA.txt";
  if (graph_name == "web-Google") return "data/web-Google.txt";
  if (graph_name == "com-lj" || graph_name == "com-LiveJournal") return "data/com-lj.ungraph.txt";
  if (graph_name == "com-orkut" || graph_name == "com-Orkut") return "data/com-orkut.ungraph.txt";
  if (graph_name == "roadNet-PA") return "data/roadNet-PA.txt";
  if (graph_name == "com-amazon") return "data/com-amazon.ungraph.txt";
  if (graph_name == "com-dblp") return "data/com-dblp.ungraph.txt";
  return "data/" + graph_name + ".txt";
}

bool BenchmarkRunner::initialize(std::string* error_msg) {
  // Check git tree cleanliness
  if (is_git_tree_dirty() && !config_.allow_dirty) {
    if (error_msg) *error_msg = "Git working tree is dirty! Commit changes or pass --allow-dirty to proceed.";
    return false;
  }

  // Load frozen thresholds
  try {
    thresholds_ = load_thresholds(config_.thresholds_path);
  } catch (const std::exception& e) {
    if (error_msg) *error_msg = "Failed to load thresholds from " + config_.thresholds_path + ": " + e.what();
    return false;
  }

  // Load or generate manifest
  std::ifstream mf_check(config_.manifest_path);
  if (mf_check.good()) {
    mf_check.close();
    manifest_ = load_manifest_json(config_.manifest_path);
  } else {
    mf_check.close();
    if (config_.dry_run) {
      manifest_ = generate_dry_run_manifest();
    } else {
      manifest_ = generate_evaluation_manifest();
    }
    save_manifest_json(manifest_, config_.manifest_path);
  }

  // Validate manifest
  size_t expected = config_.expected_manifest_count > 0 ? config_.expected_manifest_count : (config_.dry_run ? 8 : 480);
  if (!validate_manifest(manifest_, error_msg, expected, config_.dry_run)) {
    return false;
  }

  // Load completed runs from runs.jsonl
  completed_run_ids_ = load_completed_run_ids(config_.runs_jsonl_path);

  // Clean partial rows in CSV
  cleanup_partial_csv_rows(config_.csv_output_path, completed_run_ids_);

  // Ensure input hashes file (Requirement 3)
  if (!config_.dry_run) {
    std::ifstream in_check(config_.env_inputs_path);
    if (!in_check.good()) {
      in_check.close();
      system("mkdir -p results/env");
      std::ofstream out(config_.env_inputs_path);
      if (out.is_open()) {
        out << "# Input Dataset Hashes (SNAP download or R-MAT parameters)\n";
        std::vector<std::pair<std::string, std::string>> inputs = {
            {"roadNet-CA", "data/roadNet-CA.txt"},
            {"web-Google", "data/web-Google.txt"},
            {"com-lj", "data/com-lj.ungraph.txt"},
            {"com-orkut", "data/com-orkut.ungraph.txt"},
            {"rmat_a0.25", "scale=20,edge_factor=16,seed=12345,a=0.25,b=0.25,c=0.25,d=0.25"},
            {"rmat_a0.45", "scale=20,edge_factor=16,seed=12345,a=0.45,b=0.18333333333333335,c=0.18333333333333335,d=0.18333333333333335"},
            {"rmat_a0.55", "scale=20,edge_factor=16,seed=12345,a=0.55,b=0.15,c=0.15,d=0.15"},
            {"rmat_a0.65", "scale=20,edge_factor=16,seed=12345,a=0.65,b=0.11666666666666667,c=0.11666666666666667,d=0.11666666666666667"}
        };
        for (const auto& item : inputs) {
          bool is_rmat = (item.second.find("scale=") == 0);
          std::string h = compute_sha256(item.second, is_rmat);
          out << item.first << " " << h << "\n";
        }
      }
    }
  }

  return true;
}

void BenchmarkRunner::load_or_cache_graph(const std::string& graph_name) {
  if (cached_graph_name_ == graph_name && cached_graph_.n > 0) {
    return;
  }

  std::string path = resolve_graph_path(graph_name);
  GraphBuildStats gbs;
  cached_graph_ = load_snap_graph(path, gbs);
  cached_graph_name_ = graph_name;

  // Compute reference BFS distances for correctness verification
  cached_ref_dist_ = bfs_ref(cached_graph_, 0);

  // Precompute oracle modes for this graph if not present
  if (oracle_modes_cache_.find(graph_name) == oracle_modes_cache_.end()) {
    compute_oracle_modes_for_graph(graph_name);
  }
}

void BenchmarkRunner::compute_oracle_modes_for_graph(const std::string& graph_name) {
  Params p;
  p.threads = config_.threads;
  p.alpha = thresholds_.alpha;
  p.GPU_MIN_BATCH = config_.gpu_min_batch;

  std::vector<LevelLog> logs_seq, logs_static, logs_dynamic;
  bfs_run(cached_graph_, 0, Mode::SEQ, p, &logs_seq);
  bfs_run(cached_graph_, 0, Mode::STATIC, p, &logs_static);
  bfs_run(cached_graph_, 0, Mode::DYNAMIC, p, &logs_dynamic);

  size_t num_levels = logs_seq.size();
  std::vector<Mode> best_modes(num_levels, Mode::STATIC);

  for (size_t l = 0; l < num_levels; ++l) {
    double t_seq = (l < logs_seq.size()) ? logs_seq[l].time_us : 1e12;
    double t_static = (l < logs_static.size()) ? logs_static[l].time_us : 1e12;
    double t_dynamic = (l < logs_dynamic.size()) ? logs_dynamic[l].time_us : 1e12;

    if (t_seq <= t_static && t_seq <= t_dynamic) {
      best_modes[l] = Mode::SEQ;
    } else if (t_static <= t_dynamic) {
      best_modes[l] = Mode::STATIC;
    } else {
      best_modes[l] = Mode::DYNAMIC;
    }
  }

  oracle_modes_cache_[graph_name] = std::move(best_modes);
}

bool BenchmarkRunner::execute_run(const BenchmarkRun& run, std::string* error_msg) {
  load_or_cache_graph(run.graph);

  Params p;
  p.threads = config_.threads;
  p.alpha = thresholds_.alpha;
  p.lambda = thresholds_.lambda;
  p.C_max = thresholds_.C_max;
  p.C_min = thresholds_.C_min;
  p.GPU_MIN_BATCH = thresholds_.GPU_MIN_BATCH;

  // Unrecorded warm-ups (2 runs per graph & policy)
  auto warmup_key = std::make_pair(run.graph, run.policy);
  while (warmup_tracker_[warmup_key] < static_cast<int>(config_.warmups_per_policy)) {
    if (run.policy == "Adaptive") {
      bfs_adaptive(cached_graph_, 0, p, thresholds_, nullptr);
    } else if (run.policy == "Pure-Static") {
      bfs_run(cached_graph_, 0, Mode::STATIC, p, nullptr);
    } else if (run.policy == "Pure-Dynamic") {
      bfs_run(cached_graph_, 0, Mode::DYNAMIC, p, nullptr);
    } else if (run.policy == "Oracle-Replay") {
      bfs_adaptive(cached_graph_, 0, p, thresholds_, nullptr);
    }
    warmup_tracker_[warmup_key]++;
  }

  // Recorded run
  std::vector<LevelLog> logs;
  std::vector<int32_t> got;

  auto t_start = std::chrono::high_resolution_clock::now();

  if (run.policy == "Adaptive") {
    got = bfs_adaptive(cached_graph_, 0, p, thresholds_, &logs);
  } else if (run.policy == "Pure-Static") {
    got = bfs_run(cached_graph_, 0, Mode::STATIC, p, &logs);
  } else if (run.policy == "Pure-Dynamic") {
    got = bfs_run(cached_graph_, 0, Mode::DYNAMIC, p, &logs);
  } else if (run.policy == "Oracle-Replay") {
    // Dispatch level by level according to oracle precomputed fastest mode
    const auto& best_modes = oracle_modes_cache_[run.graph];
    BfsState st = make_state(cached_graph_, 0);
    GpuGraph gg = upload_graph(cached_graph_);

    while (st.frontier_size > 0) {
      uint32_t l = st.level;
      Mode m = (l < best_modes.size()) ? best_modes[l] : Mode::STATIC;

      LevelStats stats = compute_level_stats(cached_graph_, st, 0, st.frontier_size, st.level);

      LevelLog llog;
      llog.level = l;
      llog.frontier_size = stats.frontier_size;
      llog.edge_work = stats.edge_work;
      llog.mu = stats.mu;
      llog.cv = stats.cv;
      llog.dmax = stats.dmax;
      llog.threads = p.threads;
      llog.alpha = p.alpha;
      llog.gpu_min_batch = p.GPU_MIN_BATCH;

      run_level(cached_graph_, gg, m, st, 0, st.frontier_size, p, llog, stats.cv);
      logs.push_back(llog);

      std::swap(st.frontier, st.next);
      st.frontier_size = st.next_size;
      st.next_size = 0;
      st.level++;
    }

    got.assign(st.dist, st.dist + cached_graph_.n);
    free_graph(gg);
    free_state(st);
  } else {
    if (error_msg) *error_msg = "Unknown policy: " + run.policy;
    return false;
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

  if (elapsed_s > config_.timeout_sec) {
    append_run_status(config_.runs_jsonl_path, run.run_id, "timeout", run.graph, run.policy, run.round);
    if (error_msg) *error_msg = "Run timed out after " + std::to_string(elapsed_s) + "s";
    return false;
  }

  // Correctness verification against reference BFS
  Mismatch m = verify(cached_ref_dist_, got);
  bool is_correct = m.ok;

  // Open logger and write rows
  LevelLogger logger;
  logger.open(config_.csv_output_path);

  static int total_rows_written_counter = 0;
  std::string git_commit = get_build_git_commit();
  std::string driver_ver = get_driver_version();

  for (const auto& log : logs) {
    LevelRow row;
    row.run_id = run.run_id;
    row.graph = run.graph;
    row.policy = run.policy;
    row.round = run.round;
    row.level = log.level;
    row.frontier_size = log.frontier_size;
    row.edge_work = log.edge_work;
    row.mu = log.mu;
    row.cv = log.cv;
    row.dmax = log.dmax;
    row.mode_chosen = mode_to_string(log.mode_chosen);
    row.chunk_size = log.chunk_size;
    row.time_us = log.time_us;
    row.steals_attempted = log.steals_attempted;
    row.steals_succeeded = log.steals_succeeded;
    row.gpu_batches = log.gpu_batches;
    row.correct = is_correct;
    row.git_commit = git_commit;
    row.driver_version = driver_ver;
    row.threads = log.threads;
    row.alpha = log.alpha;
    row.gpu_min_batch = log.gpu_min_batch;
    row.num_chunks = log.num_chunks;
    row.gpu_batch_vertices = log.gpu_batch_vertices;
    row.t_cpu_us = log.t_cpu_us;
    row.t_gpu_us = log.t_gpu_us;
    row.t_merge_us = log.t_merge_us;
    row.gpu_idle_us = log.gpu_idle_us;

    logger.write_row(row);
    total_rows_written_counter++;

    // Testing hook: kill process if requested after writing N rows
    if (config_.kill_after_rows > 0 && total_rows_written_counter >= config_.kill_after_rows) {
      logger.close();
      raise(SIGKILL);
    }
  }

  // Commit run: fsync
  logger.end_run();
  logger.close();

  // Append completion record to runs.jsonl
  append_run_status(config_.runs_jsonl_path, run.run_id, "complete", run.graph, run.policy, run.round);
  completed_run_ids_.insert(run.run_id);

  return true;
}

RunnerSummary BenchmarkRunner::run_all() {
  RunnerSummary summary;
  summary.total_scheduled = manifest_.size();

  for (const auto& run : manifest_) {
    if (completed_run_ids_.count(run.run_id) > 0) {
      summary.runs_skipped_resumed++;
      continue;
    }

    summary.runs_executed++;
    std::string err;
    if (execute_run(run, &err)) {
      summary.runs_completed++;
    } else {
      if (err.find("timed out") != std::string::npos) {
        summary.runs_timed_out++;
      }
    }
  }

  summary.rows_written = count_csv_total_rows(config_.csv_output_path);
  return summary;
}
