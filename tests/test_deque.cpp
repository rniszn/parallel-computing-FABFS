#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "sched/deque.h"

int main(int argc, char** argv) {
  bool sanitizer_mode = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--sanitizer") {
      sanitizer_mode = true;
    }
  }

  std::cout << "==========================================" << std::endl;
  std::cout << "Running M5 Chase-Lev Deque Tests" << std::endl;
  std::cout << "==========================================" << std::endl;

  bool all_passed = true;

  // Test 1: Single thread push 1000 items, pop them all (LIFO order), steal empty
  {
    std::cout << "[Test 1] Single thread push/pop order & empty steal: ";
    ChaseLevDeque deque(2048);
    for (uint32_t i = 0; i < 1000; ++i) {
      deque.push(Range{i, i + 1});
    }

    bool order_ok = true;
    for (int i = 999; i >= 0; --i) {
      Range r;
      if (!deque.pop(r)) {
        order_ok = false;
        break;
      }
      if (r.b != static_cast<uint32_t>(i) || r.e != static_cast<uint32_t>(i + 1)) {
        order_ok = false;
        break;
      }
    }

    Range dummy;
    bool empty_pop = !deque.pop(dummy);
    bool empty_steal = !deque.steal(dummy);

    if (order_ok && empty_pop && empty_steal) {
      std::cout << "PASS (1000 items in reverse order, empty pop/steal returned false)" << std::endl;
    } else {
      std::cout << "FAIL (order_ok=" << order_ok << ", empty_pop=" << empty_pop
                << ", empty_steal=" << empty_steal << ")" << std::endl;
      all_passed = false;
    }
  }

  // Test 2: Last-item race between owner and thief
  {
    const int ROUNDS = sanitizer_mode ? 50000 : 1000000;
    std::cout << "[Test 2] Last-item race (" << ROUNDS << " rounds, 1 owner, 1 thief): ";
    ChaseLevDeque deque(16);
    std::atomic<int> owner_wins{0};
    std::atomic<int> thief_wins{0};
    std::atomic<bool> race_failed{false};

    std::atomic<bool> start_signal{false};
    std::atomic<bool> finished{false};
    std::atomic<int> current_round{0};

    // Thief thread
    std::thread thief([&]() {
      for (int r = 0; r < ROUNDS; ++r) {
        // Wait until owner has pushed
        while (current_round.load(std::memory_order_acquire) != r * 2 + 1) {
          if (finished.load(std::memory_order_relaxed)) return;
          std::this_thread::yield();
        }

        Range item;
        if (deque.steal(item)) {
          thief_wins.fetch_add(1, std::memory_order_relaxed);
        }

        // Signal thief has attempted
        current_round.store(r * 2 + 2, std::memory_order_release);
      }
    });

    // Owner thread (main)
    for (int r = 0; r < ROUNDS; ++r) {
      deque.push(Range{static_cast<uint32_t>(r), static_cast<uint32_t>(r + 1)});

      // Signal thief to race
      current_round.store(r * 2 + 1, std::memory_order_release);

      Range item;
      bool got = deque.pop(item);
      if (got) {
        owner_wins.fetch_add(1, std::memory_order_relaxed);
      }

      // Wait for thief to finish round
      while (current_round.load(std::memory_order_acquire) != r * 2 + 2) {
        std::this_thread::yield();
      }
    }

    finished.store(true, std::memory_order_release);
    thief.join();

    int total_wins = owner_wins.load() + thief_wins.load();
    if (total_wins == ROUNDS) {
      std::cout << "PASS (" << ROUNDS << " rounds: owner=" << owner_wins.load()
                << ", thief=" << thief_wins.load() << ", exactly one winner per round)" << std::endl;
    } else {
      std::cout << "FAIL (total wins=" << total_wins << " vs expected " << ROUNDS << ")" << std::endl;
      all_passed = false;
    }
  }

  // Test 3: Stress test: 8 threads, 10 million items, repeat 100 times
  {
    constexpr int NUM_THREADS = 8;
    const uint32_t NUM_ITEMS = sanitizer_mode ? 100000 : 10000000;
    const int REPEATS = sanitizer_mode ? 5 : 100;
    std::cout << "[Test 3] Stress test (8 threads, " << NUM_ITEMS << " items x " << REPEATS << " repeats): " << std::flush;

    // 8 deques
    static ChaseLevDeque deques[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; ++i) {
      deques[i].init(sanitizer_mode ? 32768 : 2097152);
    }

    std::vector<uint8_t> item_taken(NUM_ITEMS, 0);
    bool stress_ok = true;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int rep = 0; rep < REPEATS; ++rep) {
      std::fill(item_taken.begin(), item_taken.end(), 0);
      for (int i = 0; i < NUM_THREADS; ++i) {
        deques[i].reset();
      }

      // Distribute items round-robin
      for (uint32_t i = 0; i < NUM_ITEMS; ++i) {
        deques[i % NUM_THREADS].push(Range{i, i + 1});
      }

      std::atomic<uint32_t> done_count{0};
      std::vector<std::thread> workers;
      workers.reserve(NUM_THREADS);

      for (int tid = 0; tid < NUM_THREADS; ++tid) {
        workers.emplace_back([&, tid]() {
          std::minstd_rand rng(12345 + tid * 101 + rep);
          std::uniform_int_distribution<int> dist(0, NUM_THREADS - 1);

          while (done_count.load(std::memory_order_relaxed) < NUM_ITEMS) {
            Range r;
            bool got = deques[tid].pop(r);
            if (!got) {
              // Try stealing from a random victim
              int victim = dist(rng);
              if (victim != tid) {
                got = deques[victim].steal(r);
              }
            }

            if (got) {
              item_taken[r.b] += 1;
              done_count.fetch_add(1, std::memory_order_relaxed);
            } else {
              std::this_thread::yield();
            }
          }
        });
      }

      for (auto& w : workers) {
        w.join();
      }

      // Verify every item was taken exactly once
      for (uint32_t i = 0; i < NUM_ITEMS; ++i) {
        if (item_taken[i] != 1) {
          std::cout << "\n  FAIL repeat " << rep << " at item " << i
                    << " (taken=" << static_cast<int>(item_taken[i]) << ")";
          stress_ok = false;
          break;
        }
      }

      if (!stress_ok) break;

      if ((rep + 1) % 10 == 0) {
        std::cout << (rep + 1) << "% " << std::flush;
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t1 - t0).count();

    if (stress_ok) {
      std::cout << "PASS (100/100 repeats passed, total time: " << total_sec << " s)" << std::endl;
    } else {
      std::cout << " -> FAILED" << std::endl;
      all_passed = false;
    }
  }

  std::cout << "==========================================" << std::endl;
  if (all_passed) {
    std::cout << "ALL DEQUE TESTS PASSED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
  } else {
    std::cout << "SOME DEQUE TESTS FAILED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 1;
  }
}
