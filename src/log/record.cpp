#include "record.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <io.h>
#define fsync _commit
#define fileno _fileno
#define popen _popen
#define pclose _pclose
#else
#include <unistd.h>
#endif

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#endif

constexpr const char* LevelLogger::CONTRACT_HEADER;

std::string escape_csv(const std::string& s) {
  if (s.find(',') == std::string::npos && s.find('"') == std::string::npos &&
      s.find('\n') == std::string::npos && s.find('\r') == std::string::npos) {
    return s;
  }
  std::string escaped = "\"";
  for (char c : s) {
    if (c == '"') {
      escaped += "\"\"";
    } else {
      escaped += c;
    }
  }
  escaped += "\"";
  return escaped;
}

std::string get_driver_version() {
  static std::string cached_version = "";
  if (!cached_version.empty()) {
    return cached_version;
  }

#if __has_include(<cuda_runtime.h>)
  int driver_ver = 0;
  if (cudaDriverGetVersion(&driver_ver) == cudaSuccess && driver_ver > 0) {
    cached_version = std::to_string(driver_ver);
    return cached_version;
  }
#endif

  FILE* fp = popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null", "r");
  if (fp) {
    char buf[64];
    if (fgets(buf, sizeof(buf), fp)) {
      cached_version = buf;
      while (!cached_version.empty() &&
             (cached_version.back() == '\n' || cached_version.back() == '\r' || cached_version.back() == ' ')) {
        cached_version.pop_back();
      }
    }
    pclose(fp);
  }

  if (cached_version.empty()) {
    cached_version = "unknown";
  }
  return cached_version;
}

std::string get_build_git_commit() {
  std::string commit;
#ifdef FABFS_GIT_COMMIT
  commit = FABFS_GIT_COMMIT;
#else
  FILE* fp = popen("git rev-parse HEAD 2>/dev/null", "r");
  if (fp) {
    char buf[128];
    if (fgets(buf, sizeof(buf), fp)) {
      commit = buf;
      while (!commit.empty() && (commit.back() == '\n' || commit.back() == '\r')) {
        commit.pop_back();
      }
    }
    pclose(fp);
  }
  if (commit.empty()) {
    commit = "unknown";
  }
#endif

  // Check if working tree has unstaged or staged modifications
  FILE* fp_diff = popen("git status --porcelain 2>/dev/null", "r");
  if (fp_diff) {
    char buf[128];
    bool is_dirty = false;
    while (fgets(buf, sizeof(buf), fp_diff)) {
      if (buf[0] != '?' && buf[0] != '\0') {
        is_dirty = true;
        break;
      }
    }
    pclose(fp_diff);
    if (is_dirty && commit.find("-dirty") == std::string::npos) {
      commit += "-dirty";
    }
  }

  return commit;
}

LevelLogger::LevelLogger() : fp_(nullptr), fd_(-1) {}

LevelLogger::~LevelLogger() {
  close();
}

void LevelLogger::open(const std::string& filepath) {
  close();
  filepath_ = filepath;

  // Check if file already exists
  bool file_exists = false;
  std::ifstream check_file(filepath);
  if (check_file.is_open()) {
    check_file.seekg(0, std::ios::end);
    if (check_file.tellg() > 0) {
      file_exists = true;
    }
    check_file.seekg(0, std::ios::beg);
  }

  if (file_exists) {
    std::string existing_header;
    if (std::getline(check_file, existing_header)) {
      while (!existing_header.empty() &&
             (existing_header.back() == '\r' || existing_header.back() == '\n')) {
        existing_header.pop_back();
      }
      if (existing_header != CONTRACT_HEADER) {
        throw std::runtime_error("Header mismatch in CSV log file: " + filepath);
      }
    }
    check_file.close();

    fp_ = std::fopen(filepath.c_str(), "a");
    if (!fp_) {
      throw std::runtime_error("Failed to open CSV file for appending: " + filepath);
    }
    fd_ = fileno(fp_);
  } else {
    check_file.close();
    fp_ = std::fopen(filepath.c_str(), "w");
    if (!fp_) {
      throw std::runtime_error("Failed to open CSV file for writing: " + filepath);
    }
    fd_ = fileno(fp_);

    // Write header once
    std::fprintf(fp_, "%s\n", CONTRACT_HEADER);
    std::fflush(fp_);
  }
}

void LevelLogger::write_row(const LevelRow& row) {
  if (!fp_) {
    throw std::runtime_error("Cannot write row: LevelLogger is not open");
  }

  std::fprintf(fp_,
               "%s,%s,%s,%u,%u,%zu,%llu,%.6f,%.6f,%u,%s,%u,%.2f,%llu,%llu,%u,"
               "%s,%s,%s,%d,%.4f,%u,%u,%llu,%.2f,%.2f,%.2f,%.2f\n",
               escape_csv(row.run_id).c_str(),
               escape_csv(row.graph).c_str(),
               escape_csv(row.policy).c_str(),
               row.round,
               row.level,
               row.frontier_size,
               static_cast<unsigned long long>(row.edge_work),
               row.mu,
               row.cv,
               row.dmax,
               escape_csv(row.mode_chosen).c_str(),
               row.chunk_size,
               row.time_us,
               static_cast<unsigned long long>(row.steals_attempted),
               static_cast<unsigned long long>(row.steals_succeeded),
               row.gpu_batches,
               row.correct ? "true" : "false",
               escape_csv(row.git_commit).c_str(),
               escape_csv(row.driver_version).c_str(),
               row.threads,
               row.alpha,
               row.gpu_min_batch,
               row.num_chunks,
               static_cast<unsigned long long>(row.gpu_batch_vertices),
               row.t_cpu_us,
               row.t_gpu_us,
               row.t_merge_us,
               row.gpu_idle_us);

  // Flush after every row
  std::fflush(fp_);
}

void LevelLogger::end_run() {
  if (fp_ && fd_ != -1) {
    std::fflush(fp_);
    fsync(fd_);
  }
}

void LevelLogger::close() {
  if (fp_) {
    std::fflush(fp_);
    std::fclose(fp_);
    fp_ = nullptr;
    fd_ = -1;
  }
}
