#include "deque.h"
#include <algorithm>
#include <cstdlib>
#include <new>

size_t ChaseLevDeque::round_up_pow2(size_t v) {
  if (v < 2) return 2;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
#if defined(__LP64__) || defined(_WIN64)
  v |= v >> 32;
#endif
  v++;
  return v;
}

ChaseLevDeque::ChaseLevDeque(size_t min_capacity) {
  init(min_capacity);
}

ChaseLevDeque::~ChaseLevDeque() {
  delete[] buffer_;
  buffer_ = nullptr;
}

void ChaseLevDeque::init(size_t min_capacity) {
  size_t cap = round_up_pow2(std::max<size_t>(min_capacity, 2));
  if (cap != capacity_ || buffer_ == nullptr) {
    delete[] buffer_;
    capacity_ = cap;
    mask_ = static_cast<int64_t>(capacity_ - 1);
    buffer_ = new Range[capacity_];
  }
  reset();
}

void ChaseLevDeque::reset() {
  top_.store(0, std::memory_order_relaxed);
  bottom_.store(0, std::memory_order_relaxed);
}

void ChaseLevDeque::push(Range r) {
  int64_t b = bottom_.load(std::memory_order_seq_cst);
  int64_t t = top_.load(std::memory_order_seq_cst);
  if (b - t >= static_cast<int64_t>(capacity_)) {
    throw std::overflow_error("Chase-Lev deque overflow: capacity exceeded");
  }
  buffer_[b & mask_] = r;
  bottom_.store(b + 1, std::memory_order_seq_cst);
}

bool ChaseLevDeque::pop(Range& r) {
  int64_t b = bottom_.load(std::memory_order_seq_cst) - 1;
  bottom_.store(b, std::memory_order_seq_cst);
  int64_t t = top_.load(std::memory_order_seq_cst);

  if (t <= b) {
    r = buffer_[b & mask_];
    if (t == b) {
      int64_t expected = t;
      if (!top_.compare_exchange_strong(expected, t + 1,
                                        std::memory_order_seq_cst,
                                        std::memory_order_seq_cst)) {
        bottom_.store(b + 1, std::memory_order_seq_cst);
        return false;
      }
      bottom_.store(b + 1, std::memory_order_seq_cst);
      return true;
    }
    return true;
  } else {
    bottom_.store(b + 1, std::memory_order_seq_cst);
    return false;
  }
}

bool ChaseLevDeque::steal(Range& r) {
  int64_t t = top_.load(std::memory_order_seq_cst);
  int64_t b = bottom_.load(std::memory_order_seq_cst);

  if (t < b) {
    r = buffer_[t & mask_];
    int64_t expected = t;
    if (top_.compare_exchange_strong(expected, t + 1,
                                     std::memory_order_seq_cst,
                                     std::memory_order_seq_cst)) {
      return true;
    }
  }
  return false;
}

bool ChaseLevDeque::empty() const {
  int64_t b = bottom_.load(std::memory_order_relaxed);
  int64_t t = top_.load(std::memory_order_relaxed);
  return b <= t;
}

size_t ChaseLevDeque::size() const {
  int64_t b = bottom_.load(std::memory_order_relaxed);
  int64_t t = top_.load(std::memory_order_relaxed);
  return (b > t) ? static_cast<size_t>(b - t) : 0;
}
