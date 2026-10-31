#include "cost.h"
#include <algorithm>
#include <cmath>

LevelCostPrediction predict_level_cost(uint64_t edge_work, size_t frontier_size, double cv,
                                      const Thresholds& th) {
  LevelCostPrediction pred;

  double p_cpu = (th.P_cpu_edges_per_s > 0) ? static_cast<double>(th.P_cpu_edges_per_s) : 1e7;
  double p_gpu = (th.P_gpu_edges_per_s > 0) ? static_cast<double>(th.P_gpu_edges_per_s) : 5e7;
  double L = (th.L_launch_us > 0.0) ? th.L_launch_us : 15.0; // microseconds
  double C_steal_us = (th.C_steal_ns > 0.0) ? (th.C_steal_ns / 1000.0) : 0.5; // microseconds
  double alpha = (th.alpha > 0.0) ? th.alpha : (p_gpu / (p_cpu + p_gpu));

  // Sequential CPU model
  pred.t_seq_us = (static_cast<double>(edge_work) / p_cpu) * 1e6;

  // Equation (3): Static Partitioning Model
  double t_gpu = L + (alpha * static_cast<double>(edge_work) / p_gpu) * 1e6;
  double t_cpu = ((1.0 - alpha) * static_cast<double>(edge_work) / p_cpu) * 1e6;
  pred.t_static_us = std::max(t_gpu, t_cpu);

  // Equation (4): Dynamic Work-Stealing Model
  uint32_t c_l = compute_chunk_size(cv, th.C_min, th.C_max, th.lambda);
  size_t n_chunks = (frontier_size + c_l - 1) / std::max<uint32_t>(1, c_l);
  size_t n_batches = std::max<size_t>(1, (frontier_size + th.GPU_MIN_BATCH - 1) / std::max<uint32_t>(1, th.GPU_MIN_BATCH));
  double steal_factor = std::max(0.0, 1.0 - 1.0 / (1.0 + th.lambda * cv));
  double n_steals = static_cast<double>(n_chunks) * steal_factor;

  double work_time_us = (static_cast<double>(edge_work) / (p_cpu + p_gpu)) * 1e6;
  double overhead_us = static_cast<double>(n_batches) * L + n_steals * C_steal_us;
  pred.t_dynamic_us = work_time_us + overhead_us;

  // Predict mode that minimizes predicted execution time
  if (pred.t_seq_us <= pred.t_static_us && pred.t_seq_us <= pred.t_dynamic_us) {
    pred.predicted_best_mode = Mode::SEQ;
  } else if (pred.t_static_us <= pred.t_dynamic_us) {
    pred.predicted_best_mode = Mode::STATIC;
  } else {
    pred.predicted_best_mode = Mode::DYNAMIC;
  }

  return pred;
}
