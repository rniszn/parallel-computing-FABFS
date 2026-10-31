#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "sched/mode.h"

// Chase-Lev lock-free work-stealing deque with Le et al. (2013) memory orderings.
// Capacity is fixed before each level as a power of two >= chunks + 1.
// It never resizes during a level.
class ChaseLevDeque {
 public:
  explicit ChaseLevDeque(size_t min_capacity = 64);
  ~ChaseLevDeque();

  ChaseLevDeque(const ChaseLevDeque&) = delete;
  ChaseLevDeque& operator=(const ChaseLevDeque&) = delete;
  ChaseLevDeque(ChaseLevDeque&&) = delete;
  ChaseLevDeque& operator=(ChaseLevDeque&&) = delete;

  // Reset or configure capacity before a level. Capacity must be a power of two.
  void init(size_t min_capacity);

  // Reset indices to zero for a new level without reallocating if capacity is sufficient
  void reset();

  // Owner operations (invoked only by the owning thread)
  void push(Range r);
  bool pop(Range& r);

  // Thief operation (invoked concurrently by any thief thread)
  bool steal(Range& r);

  // Status queries
  bool empty() const;
  size_t size() const;
  size_t capacity() const { return capacity_; }

 private:
  static size_t round_up_pow2(size_t v);

  char pad0_[64];
  std::atomic<int64_t> top_{0};
  char pad1_[64];
  std::atomic<int64_t> bottom_{0};
  char pad2_[64];
  size_t capacity_{0};
  int64_t mask_{0};
  Range* buffer_{nullptr};
};
