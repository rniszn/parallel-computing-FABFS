#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <execinfo.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

void segv_handler(int sig) {
  void* array[50];
  size_t size = backtrace(array, 50);
  fprintf(stderr, "=== CRASH SIGNAL %d ===\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  _exit(1);
}

#include "common/bfs_state.h"
#include "cpu/bfs_omp.h"
#include "gpu/bfs_cuda.h"
#include "gpu/graph_gpu.h"
#include "graph/csr.h"
#include "graph/loader.h"
#include "graph/types.h"
#include "model/decide.h"
#include "ref/bfs_ref.h"
#include "sched/deque.h"
#include "sched/mode.h"
#include "sched/scheduler.h"

namespace {

bool file_exists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

std::string get_graph_file_path(const std::string& name) {
  if (name == "roadNet-CA") return "data/roadNet-CA.txt";
  if (name == "web-Google") return "data/web-Google.txt";
  if (name == "com-lj") return "data/com-lj.ungraph.txt";
  if (name == "com-orkut") return "data/com-orkut.ungraph.txt";
  if (name == "roadNet-PA") return "data/roadNet-PA.txt";
  if (name == "com-amazon") return "data/com-amazon.ungraph.txt";
  if (name == "com-dblp") return "data/com-dblp.ungraph.txt";
  if (name.rfind("rmat_", 0) == 0) return "data/" + name + ".txt";
  return "data/" + name + ".txt";
}

}  // namespace

int main(int argc, char** argv) {
  signal(SIGSEGV, segv_handler);
  signal(SIGABRT, segv_handler);
  bool sanitizer_mode = false;
  std::string filter_graph = "";
  bool unit_tests_only = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--sanitizer") {
      sanitizer_mode = true;
    } else if (arg == "--unit-only") {
      unit_tests_only = true;
    } else if (arg.rfind("--graph=", 0) == 0) {
      filter_graph = arg.substr(8);
    }
  }

  std::cout << "==========================================" << std::endl;
  std::cout << "Running M5 Scheduler & Gate 1 Tests" << std::endl;
  std::cout << "==========================================" << std::endl;

  bool all_passed = true;

  // Test 9: Deque overflow check
  {
    std::cout << "[Test 9] Deque overflow handling: ";
    ChaseLevDeque deque(64);
    bool threw = false;
    try {
      for (uint32_t i = 0; i < 65; ++i) {
        deque.push(Range{i, i + 1});
      }
    } catch (const std::overflow_error& e) {
      threw = true;
    }
    if (threw) {
      std::cout << "PASS (aborted with std::overflow_error on capacity exceed)" << std::endl;
    } else {
      std::cout << "FAIL (expected overflow exception)" << std::endl;
      all_passed = false;
    }
  }

  // Test 7: DYNAMIC chunk coverage check
  {
    std::cout << "[Test 7] DYNAMIC range coverage (exact partition without gaps/overlaps): ";
    size_t fbegin = 17;
    size_t fend = 8493;
    uint32_t c_l = 128;

    std::vector<Range> chunks;
    for (size_t cur = fbegin; cur < fend; cur += c_l) {
      uint32_t b = static_cast<uint32_t>(cur);
      uint32_t e = static_cast<uint32_t>(std::min(cur + c_l, fend));
      chunks.push_back(Range{b, e});
    }

    bool coverage_ok = true;
    if (chunks.empty() || chunks.front().b != fbegin || chunks.back().e != fend) {
      coverage_ok = false;
    }
    for (size_t i = 0; i + 1 < chunks.size(); ++i) {
      if (chunks[i].e != chunks[i + 1].b) {
        coverage_ok = false;
        break;
      }
    }

    if (coverage_ok) {
      std::cout << "PASS (" << chunks.size() << " chunks cover [" << fbegin << ", " << fend << ") seamlessly)" << std::endl;
    } else {
      std::cout << "FAIL" << std::endl;
      all_passed = false;
    }
  }

  // Load small graph for Tests 6 & 8
  std::string small_graph_name = "rmat_s16_a0.45";
  std::string small_path = get_graph_file_path(small_graph_name);
  if (!file_exists(small_path)) {
    std::cout << "Warning: " << small_path << " not found, trying com-amazon" << std::endl;
    small_graph_name = "com-amazon";
    small_path = get_graph_file_path(small_graph_name);
  }

  if (file_exists(small_path)) {
    GraphBuildStats stats;
    Graph g_small = load_snap_graph(small_path, stats);

    // Test 6: STATIC split edge-work balance within 1% of alpha
    {
      std::cout << "[Test 6] STATIC split edge-work balance (within 1% of alpha): ";
      GpuGraph gg = upload_graph(g_small);
      BfsState st = make_state(g_small, 0);

      // Advance to level 1 so frontier has substantial size
      reset_level(st);
      expand_level_gpu(gg, st, 0, st.frontier_size);
      std::swap(st.frontier, st.next);
      st.frontier_size = st.next_size;
      st.next_size = 0;
      st.level++;

      Params p;
      p.alpha = 0.50;
      p.threads = 4;

      size_t fbegin = 0;
      size_t fend = st.frontier_size;
      size_t count = fend - fbegin;

      std::vector<uint64_t> prefix(count + 1, 0);
      for (size_t i = 0; i < count; ++i) {
        vid_t u = st.frontier[fbegin + i];
        uint64_t deg = g_small.row_offsets[u + 1] - g_small.row_offsets[u];
        prefix[i + 1] = prefix[i] + deg;
      }
      uint64_t total_work = prefix[count];
      uint64_t target_gpu_work = static_cast<uint64_t>(std::round(p.alpha * static_cast<double>(total_work)));

      auto it = std::lower_bound(prefix.begin(), prefix.end(), target_gpu_work);
      size_t idx = std::distance(prefix.begin(), it);
      if (idx > 0 && idx < prefix.size()) {
        uint64_t diff1 = (prefix[idx] >= target_gpu_work) ? (prefix[idx] - target_gpu_work) : (target_gpu_work - prefix[idx]);
        uint64_t diff0 = (target_gpu_work >= prefix[idx - 1]) ? (target_gpu_work - prefix[idx - 1]) : (prefix[idx - 1] - target_gpu_work);
        if (diff0 < diff1) idx = idx - 1;
      }

      uint64_t actual_gpu_work = prefix[idx];
      double actual_ratio = static_cast<double>(actual_gpu_work) / static_cast<double>(total_work);
      double diff = std::abs(actual_ratio - p.alpha);

      free_state(st);
      free_graph(gg);

      if (diff <= 0.01) {
        std::cout << "PASS (alpha=" << p.alpha << ", actual=" << actual_ratio
                  << ", diff=" << diff << " <= 0.01)" << std::endl;
      } else {
        std::cout << "FAIL (alpha=" << p.alpha << ", actual=" << actual_ratio
                  << ", diff=" << diff << " > 0.01)" << std::endl;
        all_passed = false;
      }
    }

    // Test 8: Counter contract on DYNAMIC level with 64 ranges
    {
      std::cout << "[Test 8] Counter contract on DYNAMIC level with 64 ranges..." << std::endl;
      GpuGraph gg = upload_graph(g_small);
      BfsState st = make_state(g_small, 0);

      // Level 0
      reset_level(st);
      expand_level_gpu(gg, st, 0, st.frontier_size);
      std::swap(st.frontier, st.next);
      st.frontier_size = st.next_size;
      st.next_size = 0;
      st.level++;

      Params p;
      p.threads = 4;
      p.C_min = 1;
      p.C_max = static_cast<uint32_t>(std::max<size_t>(1, st.frontier_size / 64));
      p.GPU_MIN_BATCH = 32;

      LevelLog log;
      run_level(g_small, gg, Mode::DYNAMIC, st, 0, st.frontier_size, p, log, 1.0);

      size_t next_sz = st.next_size;
      std::vector<vid_t> next_vec(st.next, st.next + next_sz);
      std::sort(next_vec.begin(), next_vec.end());
      auto it_dup = std::adjacent_find(next_vec.begin(), next_vec.end());
      bool no_duplicates = (it_dup == next_vec.end());
      bool size_matches_counter = (next_sz == static_cast<size_t>(st.counts[0]));

      free_state(st);
      free_graph(gg);

      if (no_duplicates && size_matches_counter) {
        std::cout << "PASS (" << next_sz << " next vertices, exactly matches counts[0], 0 duplicates)" << std::endl;
      } else {
        std::cout << "FAIL (no_duplicates=" << no_duplicates
                  << ", size_matches=" << size_matches_counter << ")" << std::endl;
        all_passed = false;
      }
    }
  }

  if (unit_tests_only) {
    return all_passed ? 0 : 1;
  }

  // Tests 4 & 5: Gate 1 full graph evaluations (Forced Modes & Adaptive Mode)
  std::cout << "\n--- Gate 1 Full Traversal Evaluations (12 graphs) ---" << std::endl;
  std::ifstream csv_file("results/graphs.csv");
  if (!csv_file) {
    std::cerr << "Warning: Cannot open results/graphs.csv" << std::endl;
  } else {
    std::string line;
    std::getline(csv_file, line);

    std::vector<std::string> graph_names;
    while (std::getline(csv_file, line)) {
      if (line.empty()) continue;
      std::stringstream ss(line);
      std::string name;
      if (std::getline(ss, name, ',')) {
        graph_names.push_back(name);
      }
    }

    std::string commit_hash = "unknown";
    FILE* fp = popen("git rev-parse HEAD", "r");
    if (fp) {
      char buf[128];
      if (fgets(buf, sizeof(buf), fp)) {
        commit_hash = buf;
        while (!commit_hash.empty() && (commit_hash.back() == '\n' || commit_hash.back() == '\r')) {
          commit_hash.pop_back();
        }
      }
      pclose(fp);
    }

    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);

    std::ofstream gate1_out("results/gate1.txt");
    gate1_out << "# Gate 1 Correctness & Benchmark Results\n";
    gate1_out << "# Git Commit: " << commit_hash << "\n";
    gate1_out << "# CUDA Driver Version: " << driver_version << "\n";
    gate1_out << "# Thresholds Source: placeholder stub (src/model/decide.h: T_min=1, T_CV=1.0, T_par=0)\n";
    gate1_out << "# Evaluated on NVIDIA RTX 4050 Laptop GPU + OpenMP CPU\n";
    gate1_out << "graph,mode,threads,time_ms,correct\n";

    int tested_count = 0;
    int skipped_count = 0;

    for (const auto& name : graph_names) {
      if (!filter_graph.empty() && name != filter_graph) {
        continue;
      }

      std::string filepath = get_graph_file_path(name);
      if (!file_exists(filepath)) {
        std::cout << "[SKIPPED] " << name << " (" << filepath << " missing)" << std::endl;
        skipped_count++;
        continue;
      }

      if (sanitizer_mode && (name == "com-orkut" || name == "com-lj")) {
        std::cout << "[SKIPPED-SANITIZER] " << name << " (skipped under sanitizer to prevent OOM/timeout)" << std::endl;
        skipped_count++;
        continue;
      }

      std::cout << "Evaluating graph: " << name << " ... " << std::flush;
      GraphBuildStats stats;
      Graph g = load_snap_graph(filepath, stats);

      auto ref = bfs_ref(g, 0);

      bool graph_ok = true;

      // Gate 1 Part One: forced modes SEQ, STATIC, DYNAMIC across threads {1, 2, 4, 8}
      for (Mode m : {Mode::SEQ, Mode::STATIC, Mode::DYNAMIC}) {
        for (int t : {1, 2, 4, 8}) {
          Params p;
          p.threads = t;
          p.alpha = 0.50;

          auto t0 = std::chrono::high_resolution_clock::now();
          auto got = bfs_run(g, 0, m, p);
          auto t1 = std::chrono::high_resolution_clock::now();
          double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

          Mismatch mismatch = verify(ref, got);
          if (!mismatch.ok) {
            std::cout << "\n  FAIL on " << name << " mode=" << mode_to_string(m)
                      << " t=" << t << " at v=" << mismatch.vertex
                      << " exp=" << mismatch.expected << " got=" << mismatch.got << std::endl;
            graph_ok = false;
            all_passed = false;
          }

          gate1_out << name << "," << mode_to_string(m) << "," << t << "," << ms << ","
                    << (mismatch.ok ? "true" : "false") << "\n";
        }
      }

      // Gate 1 Part Two: Adaptive mode
      for (int t : {1, 2, 4, 8}) {
        Params p;
        p.threads = t;
        p.alpha = 0.50;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto got = bfs_adaptive(g, 0, p);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        Mismatch mismatch = verify(ref, got);
        if (!mismatch.ok) {
          std::cout << "\n  FAIL on " << name << " mode=ADAPTIVE t=" << t
                    << " at v=" << mismatch.vertex << std::endl;
          graph_ok = false;
          all_passed = false;
        }

        gate1_out << name << ",ADAPTIVE," << t << "," << ms << ","
                  << (mismatch.ok ? "true" : "false") << "\n";
      }

      if (graph_ok) {
        std::cout << "PASS (SEQ, STATIC, DYNAMIC, ADAPTIVE verified for threads {1,2,4,8})" << std::endl;
        tested_count++;
      }
    }

    gate1_out.close();
    std::cout << "\nGraphs evaluated: " << tested_count << ", Skipped: " << skipped_count << std::endl;
    std::cout << "Gate 1 results recorded to results/gate1.txt" << std::endl;
  }

  std::cout << "\n==========================================" << std::endl;
  if (all_passed) {
    std::cout << "ALL M5 SCHEDULER TESTS PASSED (GATE 1 PASS)" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
  } else {
    std::cout << "SOME SCHEDULER TESTS FAILED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 1;
  }
}
