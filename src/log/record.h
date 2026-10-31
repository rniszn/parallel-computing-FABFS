#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Exact 28-column contract matching FABFS Project Plan & M7 specification
struct LevelRow {
  // Shared columns (19)
  std::string run_id;
  std::string graph;
  std::string policy;
  uint32_t round{0};
  uint32_t level{0};
  size_t frontier_size{0};
  uint64_t edge_work{0};
  double mu{0.0};
  double cv{0.0};
  uint32_t dmax{0};
  std::string mode_chosen;
  uint32_t chunk_size{0};
  double time_us{0.0};
  uint64_t steals_attempted{0};
  uint64_t steals_succeeded{0};
  uint32_t gpu_batches{0};
  bool correct{true};
  std::string git_commit;
  std::string driver_version;

  // Extra columns (9)
  int threads{1};
  double alpha{0.5};
  uint32_t gpu_min_batch{4096};
  uint32_t num_chunks{0};
  uint64_t gpu_batch_vertices{0};
  double t_cpu_us{0.0};
  double t_gpu_us{0.0};
  double t_merge_us{0.0};
  double gpu_idle_us{0.0};
};

class LevelLogger {
 public:
  static constexpr const char* CONTRACT_HEADER =
      "run_id,graph,policy,round,level,frontier_size,edge_work,mu,cv,dmax,"
      "mode_chosen,chunk_size,time_us,steals_attempted,steals_succeeded,"
      "gpu_batches,correct,git_commit,driver_version,threads,alpha,"
      "gpu_min_batch,num_chunks,gpu_batch_vertices,t_cpu_us,t_gpu_us,"
      "t_merge_us,gpu_idle_us";

  LevelLogger();
  ~LevelLogger();

  LevelLogger(const LevelLogger&) = delete;
  LevelLogger& operator=(const LevelLogger&) = delete;

  // Opens file for writing. If existing, validates header matches contract.
  void open(const std::string& filepath);

  // Writes one level row. Flushes to OS page cache with fflush().
  void write_row(const LevelRow& row);

  // Commits a full run to disk by calling fsync() on the file descriptor.
  void end_run();

  // Closes the file.
  void close();

  bool is_open() const { return fp_ != nullptr; }

 private:
  FILE* fp_{nullptr};
  int fd_{-1};
  std::string filepath_;
};

// Returns git commit hash, appending "-dirty" if the working tree has uncommitted changes
std::string get_build_git_commit();

// Returns NVIDIA / CUDA driver version
std::string get_driver_version();

// Quotes / escapes string if it contains commas or quote characters
std::string escape_csv(const std::string& s);
