#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "graph/types.h"
#include "sched/mode.h"

// Online Welford accumulator for computing degree mean and M2 (sum of squared deviations)
struct WelfordAccumulator {
  uint64_t count{0};
  double mean{0.0};
  double M2{0.0};
  uint32_t dmax{0};
  uint64_t sum_deg{0};

  void update(uint32_t x) {
    count++;
    sum_deg += x;
    double delta = static_cast<double>(x) - mean;
    mean += delta / static_cast<double>(count);
    double delta2 = static_cast<double>(x) - mean;
    M2 += delta * delta2;
    if (x > dmax) {
      dmax = x;
    }
  }

  // Merges two accumulators using Chan et al. (1979 / 1982) parallel formula
  static WelfordAccumulator merge(const WelfordAccumulator& a, const WelfordAccumulator& b) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;

    WelfordAccumulator res;
    res.count = a.count + b.count;
    res.sum_deg = a.sum_deg + b.sum_deg;
    res.dmax = (a.dmax > b.dmax) ? a.dmax : b.dmax;

    double delta = b.mean - a.mean;
    res.mean = a.mean + delta * (static_cast<double>(b.count) / static_cast<double>(res.count));
    res.M2 = a.M2 + b.M2 + delta * delta *
             (static_cast<double>(a.count) * static_cast<double>(b.count) / static_cast<double>(res.count));
    return res;
  }
};

// Computes frontier statistics (|F|, edge_work, mu, cv, dmax) in one OpenMP pass
// using per-thread Welford accumulators and Chan's parallel merge formula.
LevelStats compute_frontier_stats(const Graph& g, const vid_t* frontier, size_t fbegin, size_t fend,
                                 uint32_t level = 0, int threads = 4);
