#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include "sched/mode.h"

// Thresholds for Algorithm 1 and runtime configuration
struct Thresholds {
  size_t T_min{1};
  double T_CV{1.0};
  size_t T_par{0};
  double alpha{0.5};
  uint64_t P_cpu_edges_per_s{0};
  uint64_t P_gpu_edges_per_s{0};
  double L_launch_us{0.0};
  double C_steal_ns{0.0};
  uint32_t C_max{4096};
  uint32_t C_min{32};
  double lambda{1.5};
  uint32_t GPU_MIN_BATCH{4096};
  std::string calibration_commit{"PENDING"};
  std::string status{"placeholder"};
};

// Algorithm 1 decision rule:
//   if |F| < T_min: return SEQ
//   if CV <= T_CV or |F| <= T_par: return STATIC
//   else: return DYNAMIC
// decide is a pure function. The same inputs always give the same mode.
Mode decide(size_t frontier_size, double cv, const Thresholds& thresholds = Thresholds{});

// Chunk size formula:
//   C_l = clamp_pow2(C_max / (1 + lambda * CV), C_min, C_max)
uint32_t compute_chunk_size(double cv, uint32_t c_min = 32, uint32_t c_max = 4096, double lambda = 1.5);

// Clamps value to nearest power-of-two in [c_min, c_max]
uint32_t clamp_pow2(double val, uint32_t c_min, uint32_t c_max);

// Loads thresholds from config/thresholds.json
Thresholds load_thresholds(const std::string& filepath = "config/thresholds.json");
