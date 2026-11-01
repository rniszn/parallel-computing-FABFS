#include <iostream>
#include <string>

#include "bench/manifest.h"
#include "bench/resume.h"
#include "bench/runner.h"

int main(int argc, char** argv) {
  RunnerConfig config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--dry-run") {
      config.dry_run = true;
    } else if (arg == "--allow-dirty") {
      config.allow_dirty = true;
    } else if (arg.find("--manifest=") == 0) {
      config.manifest_path = arg.substr(11);
    } else if (arg.find("--output=") == 0) {
      config.csv_output_path = arg.substr(9);
    } else if (arg.find("--runs=") == 0) {
      config.runs_jsonl_path = arg.substr(7);
    } else if (arg.find("--kill-after-rows=") == 0) {
      config.kill_after_rows = std::stoi(arg.substr(18));
    } else if (arg.find("--expected-count=") == 0) {
      config.expected_manifest_count = std::stoul(arg.substr(17));
    }
  }

  BenchmarkRunner runner(config);
  std::string err;
  if (!runner.initialize(&err)) {
    std::cerr << "Runner initialization failed: " << err << std::endl;
    return 1;
  }

  std::cout << "Starting FABFS Benchmark Execution Matrix..." << std::endl;
  RunnerSummary summary = runner.run_all();

  std::cout << "==========================================" << std::endl;
  std::cout << "Execution Summary:" << std::endl;
  std::cout << "  Total Scheduled: " << summary.total_scheduled << std::endl;
  std::cout << "  Resumed/Skipped: " << summary.runs_skipped_resumed << std::endl;
  std::cout << "  Runs Executed:   " << summary.runs_executed << std::endl;
  std::cout << "  Runs Completed:  " << summary.runs_completed << std::endl;
  std::cout << "  Timeouts:        " << summary.runs_timed_out << std::endl;
  std::cout << "  Rows Written:    " << summary.rows_written << std::endl;
  std::cout << "==========================================" << std::endl;

  return 0;
}
