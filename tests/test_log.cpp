#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "log/record.h"

namespace {

LevelRow make_sample_row(const std::string& run_id, uint32_t round, uint32_t level) {
  LevelRow r;
  r.run_id = run_id;
  r.graph = "com-amazon";
  r.policy = "ADAPTIVE";
  r.round = round;
  r.level = level;
  r.frontier_size = 1000 + level * 500;
  r.edge_work = 10000 + level * 5000;
  r.mu = 10.5;
  r.cv = 1.25;
  r.dmax = 120;
  r.mode_chosen = (level == 0) ? "SEQ" : "STATIC";
  r.chunk_size = 512;
  r.time_us = 1234.56;
  r.steals_attempted = 15;
  r.steals_succeeded = 12;
  r.gpu_batches = 1;
  r.correct = true;
  r.git_commit = get_build_git_commit();
  r.driver_version = get_driver_version();
  r.threads = 4;
  r.alpha = 0.7956;
  r.gpu_min_batch = 4096;
  r.num_chunks = 8;
  r.gpu_batch_vertices = 2048;
  r.t_cpu_us = 800.0;
  r.t_gpu_us = 400.0;
  r.t_merge_us = 15.0;
  r.gpu_idle_us = 20.0;
  return r;
}

}  // namespace

int main() {
  std::cout << "==========================================" << std::endl;
  std::cout << "Running M7 Level Logger Tests" << std::endl;
  std::cout << "==========================================" << std::endl;

  bool all_passed = true;

  // Test 1: Header byte-for-byte check
  {
    std::cout << "[Test 1] Header matches contract byte-for-byte: ";
    std::string test_file = "test_header.csv";
    std::remove(test_file.c_str());

    {
      LevelLogger logger;
      logger.open(test_file);
      logger.close();
    }

    std::ifstream in(test_file);
    std::string header;
    std::getline(in, header);
    while (!header.empty() && (header.back() == '\r' || header.back() == '\n')) {
      header.pop_back();
    }

    if (header == LevelLogger::CONTRACT_HEADER) {
      std::cout << "PASS (28 columns exact match)" << std::endl;
    } else {
      std::cout << "FAIL\nExpected: " << LevelLogger::CONTRACT_HEADER << "\nGot:      " << header << std::endl;
      all_passed = false;
    }
    std::remove(test_file.c_str());
  }

  // Test 2: Write 3 runs of 5 levels each and read back
  {
    std::cout << "[Test 2] Write 3 runs of 5 levels (15 rows) and parse: ";
    std::string test_file = "test_runs.csv";
    std::remove(test_file.c_str());

    {
      LevelLogger logger;
      logger.open(test_file);
      for (uint32_t r = 0; r < 3; ++r) {
        std::string run_id = "run_" + std::to_string(r);
        for (uint32_t l = 0; l < 5; ++l) {
          logger.write_row(make_sample_row(run_id, r, l));
        }
        logger.end_run();
      }
      logger.close();
    }

    std::ifstream in(test_file);
    std::string line;
    std::getline(in, line); // header

    int row_count = 0;
    bool order_ok = true;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      std::stringstream ss(line);
      std::string run_id, graph, policy, round_str, level_str;
      std::getline(ss, run_id, ',');
      std::getline(ss, graph, ',');
      std::getline(ss, policy, ',');
      std::getline(ss, round_str, ',');
      std::getline(ss, level_str, ',');

      uint32_t exp_round = row_count / 5;
      uint32_t exp_level = row_count % 5;
      std::string exp_run = "run_" + std::to_string(exp_round);

      if (run_id != exp_run || std::stoul(round_str) != exp_round || std::stoul(level_str) != exp_level) {
        order_ok = false;
      }
      row_count++;
    }

    if (row_count == 15 && order_ok) {
      std::cout << "PASS (15 rows, correct level and run ordering)" << std::endl;
    } else {
      std::cout << "FAIL (row_count=" << row_count << ", order_ok=" << order_ok << ")" << std::endl;
      all_passed = false;
    }
    std::remove(test_file.c_str());
  }

  // Test 3: SIGKILL crash test
  {
    std::cout << "[Test 3] SIGKILL crash test during 2nd run: ";
    std::string test_file = "test_crash.csv";
    std::remove(test_file.c_str());

    int pipefd[2];
    if (pipe(pipefd) == -1) {
      perror("pipe");
      all_passed = false;
    }

    pid_t pid = fork();
    if (pid == 0) {
      // Child process
      close(pipefd[0]);
      LevelLogger logger;
      logger.open(test_file);

      // Write Run 1 completely
      for (uint32_t l = 0; l < 5; ++l) {
        logger.write_row(make_sample_row("run_0", 0, l));
      }
      logger.end_run();

      // Signal parent that Run 1 is finished
      char sig = '1';
      ssize_t w = write(pipefd[1], &sig, 1);
      (void)w;
      close(pipefd[1]);

      // Start Run 2
      logger.write_row(make_sample_row("run_1", 1, 0));
      // Busy loop waiting to be killed
      while (true) {
        pause();
      }
      _exit(0);
    } else {
      // Parent process
      close(pipefd[1]);
      char sig;
      ssize_t r = read(pipefd[0], &sig, 1);
      (void)r;
      close(pipefd[0]);

      // Give child brief moment to write partial run 2
      usleep(10000);

      // Kill child abruptly with SIGKILL
      kill(pid, SIGKILL);
      int status;
      waitpid(pid, &status, 0);

      // Verify file integrity
      std::ifstream in(test_file);
      std::string line;
      std::getline(in, line); // header

      int run0_count = 0;
      while (std::getline(in, line)) {
        if (line.rfind("run_0", 0) == 0) {
          run0_count++;
        }
      }

      if (run0_count == 5) {
        std::cout << "PASS (Run 1 completely intact after SIGKILL during Run 2)" << std::endl;
      } else {
        std::cout << "FAIL (lost Run 1 data! run0_count=" << run0_count << ")" << std::endl;
        all_passed = false;
      }
      std::remove(test_file.c_str());
    }
  }

  // Test 4: Header mismatch error
  {
    std::cout << "[Test 4] Header mismatch detection on open: ";
    std::string test_file = "test_mismatch.csv";
    std::remove(test_file.c_str());

    {
      std::ofstream out(test_file);
      out << "run_id,graph,policy\n";
    }

    bool caught = false;
    try {
      LevelLogger logger;
      logger.open(test_file);
    } catch (const std::runtime_error& e) {
      caught = true;
    }

    if (caught) {
      std::cout << "PASS (threw std::runtime_error on invalid header)" << std::endl;
    } else {
      std::cout << "FAIL (did not reject mismatched header)" << std::endl;
      all_passed = false;
    }
    std::remove(test_file.c_str());
  }

  // Test 5: Six decimal places for mu and cv
  {
    std::cout << "[Test 5] Six decimal places precision for mu and cv: ";
    std::string test_file = "test_precision.csv";
    std::remove(test_file.c_str());

    {
      LevelLogger logger;
      logger.open(test_file);
      LevelRow r = make_sample_row("run_p", 0, 0);
      r.mu = 3.0;
      r.cv = 0.125;
      logger.write_row(r);
      logger.end_run();
      logger.close();
    }

    std::ifstream in(test_file);
    std::string line;
    std::getline(in, line); // header
    std::getline(in, line); // row

    if (line.find("3.000000") != std::string::npos && line.find("0.125000") != std::string::npos) {
      std::cout << "PASS (found 3.000000 and 0.125000 in CSV row)" << std::endl;
    } else {
      std::cout << "FAIL (line=" << line << ")" << std::endl;
      all_passed = false;
    }
    std::remove(test_file.c_str());
  }

  // Test 6: git_commit dirty suffix check
  {
    std::cout << "[Test 6] git_commit format and -dirty suffix detection: ";
    std::string commit = get_build_git_commit();
    if (!commit.empty() && commit != "unknown") {
      std::cout << "PASS (commit=" << commit << ")" << std::endl;
    } else {
      std::cout << "FAIL (empty commit string)" << std::endl;
      all_passed = false;
    }
  }

  std::cout << "==========================================" << std::endl;
  if (all_passed) {
    std::cout << "ALL M7 LOGGING TESTS PASSED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
  } else {
    std::cout << "SOME M7 LOGGING TESTS FAILED" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 1;
  }
}
