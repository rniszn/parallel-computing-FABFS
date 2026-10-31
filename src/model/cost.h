#pragma once

#include <cstddef>
#include <cstdint>
#include "model/decide.h"
#include "sched/mode.h"

/*
 * ============================================================================
 * FABFS Level Execution Cost Model (Methodology Report Equations 3 & 4)
 * ============================================================================
 *
 * Equation (3) - Static Partitioning Model:
 *   T_static(E_l, |F_l|) = max( L + (alpha * E_l) / P_gpu, ((1 - alpha) * E_l) / P_cpu )
 *   where:
 *     alpha = P_gpu / (P_cpu + P_gpu)
 *     L = GPU kernel launch latency (seconds)
 *     P_cpu = CPU edge processing throughput (edges / second)
 *     P_gpu = GPU edge processing throughput (edges / second)
 *
 * Equation (4) - Dynamic Work-Stealing Model:
 *   T_dynamic(E_l, |F_l|, CV) = (E_l / (P_cpu + P_gpu)) + (N_batches * L) + (N_steals * C_steal)
 *   where:
 *     C_l = clamp_pow2(C_max / (1 + lambda * CV), C_min, C_max)
 *     N_chunks = ceil(|F_l| / C_l)
 *     N_batches = ceil(alpha * |F_l| / GPU_MIN_BATCH)
 *     N_steals = N_chunks * (1 - 1 / (1 + lambda * CV))
 *     C_steal = cost per successful steal (seconds)
 *
 * Sequential CPU Model:
 *   T_seq(E_l) = E_l / P_cpu
 *
 * The cost model is used for calibration, threshold search validation, and
 * sanity checks. Runtime scheduling decisions are made directly by Algorithm 1.
 * ============================================================================
 */

struct LevelCostPrediction {
  double t_seq_us{0.0};
  double t_static_us{0.0};
  double t_dynamic_us{0.0};
  Mode predicted_best_mode{Mode::SEQ};
};

LevelCostPrediction predict_level_cost(uint64_t edge_work, size_t frontier_size, double cv,
                                      const Thresholds& th);
