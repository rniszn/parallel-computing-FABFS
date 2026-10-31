#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

enum class Mode { SEQ, STATIC, DYNAMIC };

inline const char* mode_to_string(Mode m) {
  switch (m) {
    case Mode::SEQ: return "SEQ";
    case Mode::STATIC: return "STATIC";
    case Mode::DYNAMIC: return "DYNAMIC";
  }
  return "UNKNOWN";
}

struct Params {
  int threads{4};
  double alpha{0.5};
  double lambda{1.5};
  uint32_t C_max{4096};
  uint32_t C_min{32};
  uint32_t GPU_MIN_BATCH{4096};
};

struct Range {
  uint32_t b{0};
  uint32_t e{0};

  bool empty() const { return b >= e; }
  size_t size() const { return (b < e) ? static_cast<size_t>(e - b) : 0; }
};

struct LevelStats {
  uint32_t level{0};
  size_t frontier_size{0};
  uint64_t edge_work{0};
  double mu{0.0};
  double cv{0.0};
  uint32_t dmax{0};
};

struct LevelLog {
  uint32_t level{0};
  size_t frontier_size{0};
  uint64_t edge_work{0};
  double mu{0.0};
  double cv{0.0};
  uint32_t dmax{0};

  Mode mode_chosen{Mode::SEQ};
  uint32_t chunk_size{0};
  uint32_t num_chunks{0};

  uint64_t steals_attempted{0};
  uint64_t steals_succeeded{0};
  uint64_t steals_cpu_to_cpu{0};
  uint64_t steals_cpu_to_gpu{0};

  uint32_t gpu_batches{0};
  uint64_t gpu_batch_vertices{0};

  double time_us{0.0};
  double t_cpu_us{0.0};
  double t_gpu_us{0.0};
  double t_merge_us{0.0};
  double gpu_idle_us{0.0};

  int threads{4};
  double alpha{0.5};
  uint32_t gpu_min_batch{4096};
};
