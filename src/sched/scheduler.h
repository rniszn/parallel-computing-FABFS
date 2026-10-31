#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/bfs_state.h"
#include "gpu/graph_gpu.h"
#include "graph/types.h"
#include "model/decide.h"
#include "sched/mode.h"

// Computes LevelStats over frontier[fbegin..fend) in one single pass
LevelStats compute_level_stats(const Graph& g, const BfsState& st, size_t fbegin, size_t fend, uint32_t level);

// Clamps value to power-of-two between c_min and c_max
uint32_t clamp_pow2(double val, uint32_t c_min, uint32_t c_max);

// Dispatches and runs one level in the given mode
void run_level(const Graph& g, const GpuGraph& gg, Mode mode, BfsState& st,
               size_t fbegin, size_t fend, const Params& p, LevelLog& log, double cv = 0.0);

// Global context runner overload matching the M5 plan signature
void set_sched_context(const Graph* g, const GpuGraph* gg, double cv = 0.0);
void run_level(Mode mode, BfsState& st, size_t fbegin, size_t fend, const Params& p, LevelLog& log);

// Full BFS driver for a forced mode (SEQ, STATIC, DYNAMIC)
std::vector<int32_t> bfs_run(const Graph& g, vid_t src, Mode forced_mode,
                             const Params& p, std::vector<LevelLog>* logs = nullptr);

// Full BFS driver for adaptive mode (calls decide() on each level)
std::vector<int32_t> bfs_adaptive(const Graph& g, vid_t src, const Params& p,
                                  const Thresholds& th = Thresholds{},
                                  std::vector<LevelLog>* logs = nullptr);
