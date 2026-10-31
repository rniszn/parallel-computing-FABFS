#include "stats.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

LevelStats compute_frontier_stats(const Graph& g, const vid_t* frontier, size_t fbegin, size_t fend,
                                 uint32_t level, int threads) {
  LevelStats stats;
  stats.level = level;

  if (fbegin >= fend || frontier == nullptr) {
    return stats;
  }

  size_t count = fend - fbegin;
  stats.frontier_size = count;

  if (threads < 1) {
    threads = 1;
  }

  std::vector<WelfordAccumulator> thread_acc(threads);

  #pragma omp parallel num_threads(threads)
  {
    int tid = omp_get_thread_num();
    #pragma omp for schedule(static)
    for (size_t i = fbegin; i < fend; ++i) {
      vid_t u = frontier[i];
      uint32_t deg = static_cast<uint32_t>(g.row_offsets[u + 1] - g.row_offsets[u]);
      thread_acc[tid].update(deg);
    }
  }

  // Merge per-thread accumulators with Chan's parallel formula
  WelfordAccumulator total = thread_acc[0];
  for (int t = 1; t < threads; ++t) {
    total = WelfordAccumulator::merge(total, thread_acc[t]);
  }

  stats.edge_work = total.sum_deg;
  stats.mu = total.mean;
  stats.dmax = total.dmax;

  if (total.count > 0 && total.mean > 0.0) {
    // Population standard deviation: sqrt(M2 / N)
    double variance = total.M2 / static_cast<double>(total.count);
    double sigma = std::sqrt(std::max(0.0, variance));
    stats.cv = sigma / total.mean;
  } else {
    stats.cv = 0.0;
  }

  return stats;
}
