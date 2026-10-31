#include "decide.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

uint32_t clamp_pow2(double val, uint32_t c_min, uint32_t c_max) {
  if (val <= static_cast<double>(c_min)) {
    return c_min;
  }
  if (val >= static_cast<double>(c_max)) {
    return c_max;
  }
  uint32_t v = static_cast<uint32_t>(std::round(val));
  if (v <= c_min) return c_min;
  if (v >= c_max) return c_max;

  uint32_t p = 1;
  while (p * 2 <= v && p * 2 <= c_max) {
    p *= 2;
  }
  uint32_t next_p = p * 2;
  if (next_p <= c_max && (v - p) > (next_p - v)) {
    return next_p;
  }
  return p;
}

uint32_t compute_chunk_size(double cv, uint32_t c_min, uint32_t c_max, double lambda) {
  double target = static_cast<double>(c_max) / (1.0 + lambda * cv);
  return clamp_pow2(target, c_min, c_max);
}

Mode decide(size_t frontier_size, double cv, const Thresholds& thresholds) {
  if (frontier_size < thresholds.T_min) {
    return Mode::SEQ;
  }
  if (cv <= thresholds.T_CV || frontier_size <= thresholds.T_par) {
    return Mode::STATIC;
  }
  return Mode::DYNAMIC;
}

Thresholds load_thresholds(const std::string& filepath) {
  Thresholds th;
  std::ifstream f(filepath);
  if (!f.is_open()) {
    return th;
  }
  std::string line;
  while (std::getline(f, line)) {
    auto parse_val = [&](const std::string& key) -> std::string {
      size_t pos = line.find("\"" + key + "\"");
      if (pos == std::string::npos) return "";
      size_t colon = line.find(':', pos);
      if (colon == std::string::npos) return "";
      size_t start = line.find_first_not_of(" \t,\"", colon + 1);
      size_t end = line.find_last_not_of(" \t,\"\r\n");
      if (start == std::string::npos || end == std::string::npos || start > end) return "";
      return line.substr(start, end - start + 1);
    };

    std::string val;
    if (!(val = parse_val("calibration_commit")).empty()) th.calibration_commit = val;
    if (!(val = parse_val("status")).empty()) th.status = val;
    if (!(val = parse_val("T_min")).empty()) th.T_min = std::stoull(val);
    if (!(val = parse_val("T_CV")).empty()) th.T_CV = std::stod(val);
    if (!(val = parse_val("T_par")).empty()) th.T_par = std::stoull(val);
    if (!(val = parse_val("alpha")).empty()) th.alpha = std::stod(val);
    if (!(val = parse_val("P_cpu_edges_per_s")).empty()) th.P_cpu_edges_per_s = std::stoull(val);
    if (!(val = parse_val("P_gpu_edges_per_s")).empty()) th.P_gpu_edges_per_s = std::stoull(val);
    if (!(val = parse_val("L_launch_us")).empty()) th.L_launch_us = std::stod(val);
    if (!(val = parse_val("C_steal_ns")).empty()) th.C_steal_ns = std::stod(val);
    if (!(val = parse_val("C_max")).empty()) th.C_max = static_cast<uint32_t>(std::stoul(val));
    if (!(val = parse_val("C_min")).empty()) th.C_min = static_cast<uint32_t>(std::stoul(val));
    if (!(val = parse_val("lambda")).empty()) th.lambda = std::stod(val);
    if (!(val = parse_val("GPU_MIN_BATCH")).empty()) th.GPU_MIN_BATCH = static_cast<uint32_t>(std::stoul(val));
  }
  return th;
}
