#include "gpu_proxy.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <random>
#include <thread>
#include <vector>

std::mutex g_managed_mem_mutex;

bool is_concurrent_managed() {
  static int concurrent = -1;
  if (concurrent == -1) {
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess) {
      cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, dev);
    } else {
      concurrent = 0;
    }
  }
  return concurrent == 1;
}

void run_gpu_proxy_dynamic(
    const GpuGraph& gg,
    BfsState& st,
    std::vector<ChaseLevDeque*>& cpu_deques,
    ChaseLevDeque* gpu_deque,
    std::atomic<uint32_t>& done_chunks,
    uint32_t total_chunks,
    uint32_t gpu_min_batch,
    LevelLog& log) {
  size_t num_cpu_deques = cpu_deques.size();
  if (num_cpu_deques == 0 || total_chunks == 0) {
    return;
  }

  std::minstd_rand rng(98765);
  std::uniform_int_distribution<size_t> dist_victim(0, num_cpu_deques - 1);

  uint64_t proxy_steals_attempted = 0;
  uint64_t proxy_steals_succeeded = 0;
  uint32_t proxy_batches = 0;
  uint64_t proxy_batch_vertices = 0;
  double proxy_gpu_us = 0.0;
  double proxy_merge_us = 0.0;
  double proxy_idle_us = 0.0;

  auto t_last_active = std::chrono::high_resolution_clock::now();

  std::vector<Range> pending_ranges;
  pending_ranges.reserve(32);

  while (done_chunks.load(std::memory_order_relaxed) < total_chunks) {
    // 1. Try to pop from gpu_deque first if anything was queued
    Range r;
    bool got = false;
    if (gpu_deque != nullptr) {
      got = gpu_deque->pop(r);
    }

    // 2. Steal from CPU deques if not found
    if (!got) {
      proxy_steals_attempted++;
      size_t start_victim = dist_victim(rng);
      for (size_t i = 0; i < num_cpu_deques; ++i) {
        size_t victim = (start_victim + i) % num_cpu_deques;
        if (cpu_deques[victim]->steal(r)) {
          got = true;
          proxy_steals_succeeded++;
          break;
        }
      }
    }

    if (!got) {
      auto t_now = std::chrono::high_resolution_clock::now();
      proxy_idle_us += std::chrono::duration<double, std::micro>(t_now - t_last_active).count();
      std::this_thread::yield();
      t_last_active = std::chrono::high_resolution_clock::now();
      continue;
    }

    // Got a range
    pending_ranges.clear();
    pending_ranges.push_back(r);
    size_t batch_vertices = r.size();
    uint32_t absorbed_chunks = 1;

    // 3. Try to merge adjacent ranges up to gpu_min_batch
    auto t_merge_start = std::chrono::high_resolution_clock::now();
    while (batch_vertices < gpu_min_batch &&
           done_chunks.load(std::memory_order_relaxed) + absorbed_chunks < total_chunks) {
      Range next_r;
      bool got_more = false;
      if (gpu_deque != nullptr) {
        got_more = gpu_deque->pop(next_r);
      }
      if (!got_more) {
        proxy_steals_attempted++;
        size_t start_v = dist_victim(rng);
        for (size_t i = 0; i < num_cpu_deques; ++i) {
          size_t v = (start_v + i) % num_cpu_deques;
          if (cpu_deques[v]->steal(next_r)) {
            got_more = true;
            proxy_steals_succeeded++;
            break;
          }
        }
      }

      if (!got_more) {
        break;
      }

      absorbed_chunks++;

      // Check if adjacent to the last range in pending_ranges
      if (pending_ranges.back().e == next_r.b) {
        pending_ranges.back().e = next_r.e;
        batch_vertices += next_r.size();
      } else if (next_r.e == pending_ranges.front().b) {
        pending_ranges.front().b = next_r.b;
        batch_vertices += next_r.size();
      } else {
        pending_ranges.push_back(next_r);
        batch_vertices += next_r.size();
      }
    }
    auto t_merge_end = std::chrono::high_resolution_clock::now();
    proxy_merge_us += std::chrono::duration<double, std::micro>(t_merge_end - t_merge_start).count();

    // 4. Execute the batch of ranges on GPU
    bool concurrent = is_concurrent_managed();
    auto t_gpu_start = std::chrono::high_resolution_clock::now();
    if (!concurrent) {
      std::lock_guard<std::mutex> lock(g_managed_mem_mutex);
      for (const auto& pr : pending_ranges) {
        if (!pr.empty()) {
          expand_level_gpu(gg, st, pr.b, pr.e);
        }
      }
    } else {
      for (const auto& pr : pending_ranges) {
        if (!pr.empty()) {
          expand_level_gpu(gg, st, pr.b, pr.e);
        }
      }
    }
    auto t_gpu_end = std::chrono::high_resolution_clock::now();
    proxy_gpu_us += std::chrono::duration<double, std::micro>(t_gpu_end - t_gpu_start).count();

    proxy_batches++;
    proxy_batch_vertices += batch_vertices;

    done_chunks.fetch_add(absorbed_chunks, std::memory_order_relaxed);
    t_last_active = std::chrono::high_resolution_clock::now();
  }

  log.steals_attempted += proxy_steals_attempted;
  log.steals_succeeded += proxy_steals_succeeded;
  log.steals_cpu_to_gpu += proxy_steals_succeeded;
  log.gpu_batches += proxy_batches;
  log.gpu_batch_vertices += proxy_batch_vertices;
  log.t_gpu_us += proxy_gpu_us;
  log.t_merge_us += proxy_merge_us;
  log.gpu_idle_us += proxy_idle_us;
}
