#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <vector>

#include "common/bfs_state.h"
#include "gpu/bfs_cuda.h"
#include "gpu/graph_gpu.h"
#include "sched/deque.h"
#include "sched/mode.h"

extern std::mutex g_managed_mem_mutex;
bool is_concurrent_managed();

// GPU Proxy Thread Worker for DYNAMIC Work-Stealing Mode.
// Steals ranges from CPU worker deques, merges adjacent ranges into batches
// of at least GPU_MIN_BATCH vertices, and calls expand_level_gpu.
void run_gpu_proxy_dynamic(
    const GpuGraph& gg,
    BfsState& st,
    std::vector<ChaseLevDeque*>& cpu_deques,
    ChaseLevDeque* gpu_deque,
    std::atomic<uint32_t>& done_chunks,
    uint32_t total_chunks,
    uint32_t gpu_min_batch,
    LevelLog& log);
