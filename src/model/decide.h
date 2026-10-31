#pragma once

#include <cstddef>
#include <cstdint>
#include "sched/mode.h"

// Placeholder thresholds struct for Module M5 before Module M6 calibration
// M6 will replace these placeholders with frozen values calibrated on calibration graphs.
struct Thresholds {
  size_t T_min{1};     // Placeholder: frontiers smaller than T_min run SEQ
  double T_CV{1.0};    // Placeholder: degree coefficient of variation threshold
  size_t T_par{0};     // Placeholder: frontiers smaller than T_par with low CV run STATIC
};

// Algorithm 1 decision rule (stub implementation for M5)
inline Mode decide(size_t frontier_size, double cv, const Thresholds& t = Thresholds{}) {
  if (frontier_size < t.T_min) {
    return Mode::SEQ;
  } else if (cv <= t.T_CV || frontier_size <= t.T_par) {
    return Mode::STATIC;
  } else {
    return Mode::DYNAMIC;
  }
}
