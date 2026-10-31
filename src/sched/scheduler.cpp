#include "scheduler.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "cpu/bfs_omp.h"
#include "gpu/bfs_cuda.h"
#include "sched/deque.h"
#include "sched/gpu_proxy.h"

namespace {
thread_local const Graph* g_ctx_graph = nullptr;
thread_local const GpuGraph* g_ctx_gpu_graph = nullptr;
thread_local double g_ctx_cv = 0.0;
}  // namespace

void set_sched_context(const Graph* g, const GpuGraph* gg, double cv) {
  g_ctx_graph = g;
  g_ctx_gpu_graph = gg;
  g_ctx_cv = cv;
}

LevelStats compute_level_stats(const Graph& g, const BfsState& st, size_t fbegin, size_t fend, uint32_t level) {
  LevelStats stats;
  stats.level = level;
  if (fbegin >= fend) {
    return stats;
  }

  size_t count = fend - fbegin;
  stats.frontier_size = count;

  uint64_t total_deg = 0;
  uint32_t max_d = 0;

  for (size_t i = fbegin; i < fend; ++i) {
    vid_t u = st.frontier[i];
    uint32_t d = static_cast<uint32_t>(g.row_offsets[u + 1] - g.row_offsets[u]);
    total_deg += d;
    if (d > max_d) max_d = d;
  }

  stats.edge_work = total_deg;
  stats.dmax = max_d;
  stats.mu = static_cast<double>(total_deg) / static_cast<double>(count);

  if (count <= 1 || stats.mu == 0.0) {
    stats.cv = 0.0;
  } else {
    double sum_sq_diff = 0.0;
    for (size_t i = fbegin; i < fend; ++i) {
      vid_t u = st.frontier[i];
      double d = static_cast<double>(g.row_offsets[u + 1] - g.row_offsets[u]);
      double diff = d - stats.mu;
      sum_sq_diff += diff * diff;
    }
    double variance = sum_sq_diff / static_cast<double>(count);
    double sigma = std::sqrt(variance);
    stats.cv = sigma / stats.mu;
  }

  return stats;
}

uint32_t clamp_pow2(double val, uint32_t c_min, uint32_t c_max) {
  if (val <= static_cast<double>(c_min)) return c_min;
  if (val >= static_cast<double>(c_max)) return c_max;
  uint32_t v = static_cast<uint32_t>(val);
  uint32_t p = 1;
  while ((p << 1) <= v) {
    p <<= 1;
  }
  if (p < c_min) p = c_min;
  if (p > c_max) p = c_max;
  return p;
}

void run_level(Mode mode, BfsState& st, size_t fbegin, size_t fend, const Params& p, LevelLog& log) {
  if (g_ctx_graph == nullptr || g_ctx_gpu_graph == nullptr) {
    throw std::runtime_error("Scheduler execution context not set");
  }
  run_level(*g_ctx_graph, *g_ctx_gpu_graph, mode, st, fbegin, fend, p, log, g_ctx_cv);
}

void run_level(const Graph& g, const GpuGraph& gg, Mode mode, BfsState& st,
               size_t fbegin, size_t fend, const Params& p, LevelLog& log, double cv) {
  // Step 1: Call reset_level once before any worker starts
  reset_level(st);

  log.mode_chosen = mode;
  log.threads = p.threads;
  log.alpha = p.alpha;
  log.gpu_min_batch = p.GPU_MIN_BATCH;

  if (fbegin >= fend) {
    st.next_size = 0;
    return;
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  if (mode == Mode::SEQ) {
    auto t0 = std::chrono::high_resolution_clock::now();
    expand_level_cpu(g, st, fbegin, fend, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    log.t_cpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    st.next_size = static_cast<size_t>(st.counts[0]);
  } else if (mode == Mode::STATIC) {
    // 1. Prefix sum of degrees over frontier[fbegin..fend)
    size_t count = fend - fbegin;
    std::vector<uint64_t> prefix(count + 1, 0);
    for (size_t i = 0; i < count; ++i) {
      vid_t u = st.frontier[fbegin + i];
      uint64_t deg = g.row_offsets[u + 1] - g.row_offsets[u];
      prefix[i + 1] = prefix[i] + deg;
    }
    uint64_t total_work = prefix[count];
    uint64_t target_gpu_work = static_cast<uint64_t>(std::round(p.alpha * static_cast<double>(total_work)));

    // 2. Binary search for split s closest to target
    auto it = std::lower_bound(prefix.begin(), prefix.end(), target_gpu_work);
    size_t idx = std::distance(prefix.begin(), it);
    if (idx > 0 && idx < prefix.size()) {
      uint64_t diff1 = (prefix[idx] >= target_gpu_work) ? (prefix[idx] - target_gpu_work) : (target_gpu_work - prefix[idx]);
      uint64_t diff0 = (target_gpu_work >= prefix[idx - 1]) ? (target_gpu_work - prefix[idx - 1]) : (prefix[idx - 1] - target_gpu_work);
      if (diff0 < diff1) {
        idx = idx - 1;
      }
    }
    size_t s = fbegin + idx;

    // 3. GPU proxy thread runs [fbegin, s), CPU runs [s, fend)
    bool concurrent = is_concurrent_managed();
    if (concurrent) {
      std::thread gpu_thread;
      auto t_gpu_start = std::chrono::high_resolution_clock::now();
      if (s > fbegin) {
        gpu_thread = std::thread([&]() {
          expand_level_gpu(gg, st, fbegin, s);
        });
        log.gpu_batches = 1;
        log.gpu_batch_vertices = (s - fbegin);
      }

      auto t_cpu_start = std::chrono::high_resolution_clock::now();
      if (s < fend) {
        expand_level_cpu(g, st, s, fend, p.threads);
      }
      auto t_cpu_end = std::chrono::high_resolution_clock::now();
      log.t_cpu_us = std::chrono::duration<double, std::micro>(t_cpu_end - t_cpu_start).count();

      if (gpu_thread.joinable()) {
        gpu_thread.join();
        auto t_gpu_end = std::chrono::high_resolution_clock::now();
        log.t_gpu_us = std::chrono::duration<double, std::micro>(t_gpu_end - t_gpu_start).count();
      }
    } else {
      // Platform adaptation for concurrentManagedAccess == 0 (e.g. WSL2):
      // Execute GPU and CPU slices sequentially to prevent invalid concurrent page faults.
      auto t_gpu_start = std::chrono::high_resolution_clock::now();
      if (s > fbegin) {
        expand_level_gpu(gg, st, fbegin, s);
        log.gpu_batches = 1;
        log.gpu_batch_vertices = (s - fbegin);
      }
      auto t_gpu_end = std::chrono::high_resolution_clock::now();
      log.t_gpu_us = std::chrono::duration<double, std::micro>(t_gpu_end - t_gpu_start).count();

      auto t_cpu_start = std::chrono::high_resolution_clock::now();
      if (s < fend) {
        expand_level_cpu(g, st, s, fend, p.threads);
      }
      auto t_cpu_end = std::chrono::high_resolution_clock::now();
      log.t_cpu_us = std::chrono::duration<double, std::micro>(t_cpu_end - t_cpu_start).count();
    }

    st.next_size = static_cast<size_t>(st.counts[0]);
  } else if (mode == Mode::DYNAMIC) {
    // 1. Chunk size C_l
    uint32_t c_l = clamp_pow2(static_cast<double>(p.C_max) / (1.0 + p.lambda * cv), p.C_min, p.C_max);
    log.chunk_size = c_l;

    // 2. Split [fbegin, fend) into chunks
    std::vector<Range> chunks;
    for (size_t cur = fbegin; cur < fend; cur += c_l) {
      uint32_t b = static_cast<uint32_t>(cur);
      uint32_t e = static_cast<uint32_t>(std::min(cur + c_l, fend));
      chunks.push_back(Range{b, e});
    }
    uint32_t num_chunks = static_cast<uint32_t>(chunks.size());
    log.num_chunks = num_chunks;

    // 3. Setup deques
    int T = std::max(1, p.threads);
    std::vector<std::unique_ptr<ChaseLevDeque>> cpu_deques;
    cpu_deques.reserve(T);
    for (int i = 0; i < T; ++i) {
      cpu_deques.push_back(std::make_unique<ChaseLevDeque>(num_chunks + 64));
    }

    ChaseLevDeque gpu_deque(num_chunks + 64);

    // Deal chunks round-robin to CPU deques
    for (size_t i = 0; i < chunks.size(); ++i) {
      cpu_deques[i % T]->push(chunks[i]);
    }

    std::vector<ChaseLevDeque*> raw_cpu_deques;
    raw_cpu_deques.reserve(T);
    for (int i = 0; i < T; ++i) {
      raw_cpu_deques.push_back(cpu_deques[i].get());
    }

    std::atomic<uint32_t> done_chunks{0};
    std::atomic<uint64_t> cpu_steals_attempted{0};
    std::atomic<uint64_t> cpu_steals_succeeded{0};
    std::atomic<uint64_t> cpu_steals_from_cpu{0};
    bool concurrent = is_concurrent_managed();

    // 4. Start GPU proxy thread
    std::thread gpu_proxy_thread([&]() {
      run_gpu_proxy_dynamic(
          gg, st, raw_cpu_deques, &gpu_deque, done_chunks, num_chunks, p.GPU_MIN_BATCH, log);
    });

    // 5. Start CPU workers
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> cpu_workers;
    cpu_workers.reserve(T);

    for (int tid = 0; tid < T; ++tid) {
      cpu_workers.emplace_back([&, tid]() {
        std::minstd_rand rng(10007 + tid * 37);
        std::uniform_int_distribution<int> dist_victim(0, T - 1);

        while (done_chunks.load(std::memory_order_relaxed) < num_chunks) {
          Range r;
          bool got = cpu_deques[tid]->pop(r);
          if (!got) {
            // Attempt steal from random victim
            cpu_steals_attempted.fetch_add(1, std::memory_order_relaxed);
            int victim = dist_victim(rng);
            if (victim != tid) {
              if (cpu_deques[victim]->steal(r)) {
                got = true;
                cpu_steals_succeeded.fetch_add(1, std::memory_order_relaxed);
                cpu_steals_from_cpu.fetch_add(1, std::memory_order_relaxed);
              }
            } else {
              // Try stealing from gpu_deque if empty
              if (gpu_deque.steal(r)) {
                got = true;
                cpu_steals_succeeded.fetch_add(1, std::memory_order_relaxed);
              }
            }
          }

          if (got) {
            if (!concurrent) {
              std::lock_guard<std::mutex> lock(g_managed_mem_mutex);
              expand_level_cpu(g, st, r.b, r.e, 1);
            } else {
              expand_level_cpu(g, st, r.b, r.e, 1);
            }
            done_chunks.fetch_add(1, std::memory_order_relaxed);
          } else {
            std::this_thread::yield();
          }
        }
      });
    }

    for (auto& cw : cpu_workers) {
      cw.join();
    }
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    log.t_cpu_us = std::chrono::duration<double, std::micro>(t_cpu_end - t_cpu_start).count();

    if (gpu_proxy_thread.joinable()) {
      gpu_proxy_thread.join();
    }

    log.steals_attempted += cpu_steals_attempted.load();
    log.steals_succeeded += cpu_steals_succeeded.load();
    log.steals_cpu_to_cpu += cpu_steals_from_cpu.load();

    st.next_size = static_cast<size_t>(st.counts[0]);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  log.time_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
}

std::vector<int32_t> bfs_run(const Graph& g, vid_t src, Mode forced_mode,
                             const Params& p, std::vector<LevelLog>* logs) {
  if (src >= g.n) {
    throw std::out_of_range("Source vertex out of range");
  }

  GpuGraph gg = upload_graph(g);
  BfsState st = make_state(g, src);

  while (st.frontier_size > 0) {
    LevelStats stats = compute_level_stats(g, st, 0, st.frontier_size, st.level);

    LevelLog log;
    log.level = stats.level;
    log.frontier_size = stats.frontier_size;
    log.edge_work = stats.edge_work;
    log.mu = stats.mu;
    log.cv = stats.cv;
    log.dmax = stats.dmax;

    run_level(g, gg, forced_mode, st, 0, st.frontier_size, p, log, stats.cv);

    if (logs != nullptr) {
      logs->push_back(log);
    }

    std::swap(st.frontier, st.next);
    st.frontier_size = st.next_size;
    st.next_size = 0;
    st.level++;
  }

  std::vector<int32_t> dist(st.dist, st.dist + g.n);
  free_state(st);
  free_graph(gg);

  return dist;
}

std::vector<int32_t> bfs_adaptive(const Graph& g, vid_t src, const Params& p,
                                  const Thresholds& th,
                                  std::vector<LevelLog>* logs) {
  if (src >= g.n) {
    throw std::out_of_range("Source vertex out of range");
  }

  GpuGraph gg = upload_graph(g);
  BfsState st = make_state(g, src);

  while (st.frontier_size > 0) {
    LevelStats stats = compute_level_stats(g, st, 0, st.frontier_size, st.level);
    Mode chosen = decide(stats.frontier_size, stats.cv, th);

    LevelLog log;
    log.level = stats.level;
    log.frontier_size = stats.frontier_size;
    log.edge_work = stats.edge_work;
    log.mu = stats.mu;
    log.cv = stats.cv;
    log.dmax = stats.dmax;

    run_level(g, gg, chosen, st, 0, st.frontier_size, p, log, stats.cv);

    if (logs != nullptr) {
      logs->push_back(log);
    }

    std::swap(st.frontier, st.next);
    st.frontier_size = st.next_size;
    st.next_size = 0;
    st.level++;
  }

  std::vector<int32_t> dist(st.dist, st.dist + g.n);
  free_state(st);
  free_graph(gg);

  return dist;
}
