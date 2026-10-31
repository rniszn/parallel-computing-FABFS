CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -std=c++17 -O3 -fopenmp -I src
NVCCFLAGS ?= -std=c++17 -O3 -Xcompiler -fopenmp -I src

# Compile-time git commit with -dirty suffix if working tree has modifications
GIT_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null)$(shell git status --porcelain 2>/dev/null | grep -E '^[^?]' >/dev/null && echo '-dirty')
DEFINES ?= -DFABFS_GIT_COMMIT=\"$(GIT_COMMIT)\"

COMMON_SRC = src/common/managed_alloc.cu
GRAPH_SRC = src/graph/csr.cpp src/graph/loader.cpp src/graph/rmat.cpp src/graph/validate.cpp
REF_SRC = src/ref/bfs_ref.cpp
CPU_SRC = src/cpu/bfs_omp.cpp
GPU_SRC = src/gpu/graph_gpu.cu src/gpu/bfs_cuda.cu
SCHED_SRC = src/sched/deque.cpp src/sched/gpu_proxy.cpp src/sched/scheduler.cpp
MODEL_SRC = src/model/stats.cpp src/model/cost.cpp src/model/decide.cpp
LOG_SRC = src/log/record.cpp
BENCH_SRC = src/bench/manifest.cpp src/bench/resume.cpp src/bench/runner.cpp

all: tests/test_log tests/test_model tests/test_sched tests/test_deque bin/calibrate_bench tests/test_bench bin/bench_runner

tests/test_log: tests/test_log.cpp $(LOG_SRC)
	$(NVCC) $(NVCCFLAGS) $(DEFINES) tests/test_log.cpp $(LOG_SRC) -o $@

tests/test_model: tests/test_model.cpp $(MODEL_SRC) src/graph/csr.cpp src/graph/loader.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) tests/test_model.cpp $(MODEL_SRC) src/graph/csr.cpp src/graph/loader.cpp -o $@

tests/test_sched: tests/test_sched.cpp $(COMMON_SRC) $(CPU_SRC) $(GPU_SRC) $(SCHED_SRC) $(MODEL_SRC) src/graph/csr.cpp src/graph/loader.cpp $(REF_SRC)
	$(NVCC) $(NVCCFLAGS) $(DEFINES) tests/test_sched.cpp $(COMMON_SRC) $(CPU_SRC) $(GPU_SRC) $(SCHED_SRC) $(MODEL_SRC) src/graph/csr.cpp src/graph/loader.cpp $(REF_SRC) -o $@

tests/test_deque: tests/test_deque.cpp src/sched/deque.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) tests/test_deque.cpp src/sched/deque.cpp -o $@

bin/calibrate_bench: tools/calibrate_bench.cu $(COMMON_SRC) $(CPU_SRC) $(GPU_SRC) $(SCHED_SRC) $(MODEL_SRC) src/graph/csr.cpp src/graph/loader.cpp $(REF_SRC)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(DEFINES) tools/calibrate_bench.cu $(COMMON_SRC) $(CPU_SRC) $(GPU_SRC) $(SCHED_SRC) $(MODEL_SRC) src/graph/csr.cpp src/graph/loader.cpp $(REF_SRC) -o $@

tests/test_bench: tests/test_bench.cpp $(COMMON_SRC) $(CPU_SRC) $(GPU_SRC) $(SCHED_SRC) $(MODEL_SRC) $(LOG_SRC) $(BENCH_SRC) src/graph/csr.cpp src/graph/loader.cpp $(REF_SRC)
	$(NVCC) $(NVCCFLAGS) $(DEFINES) tests/test_bench.cpp $(COMMON_SRC) $(CPU_SRC) $(GPU_SRC) $(SCHED_SRC) $(MODEL_SRC) $(LOG_SRC) $(BENCH_SRC) src/graph/csr.cpp src/graph/loader.cpp $(REF_SRC) -o $@

bin/bench_runner: tools/bench_runner.cpp $(COMMON_SRC) $(CPU_SRC) $(GPU_SRC) $(SCHED_SRC) $(MODEL_SRC) $(LOG_SRC) $(BENCH_SRC) src/graph/csr.cpp src/graph/loader.cpp $(REF_SRC)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(DEFINES) tools/bench_runner.cpp $(COMMON_SRC) $(CPU_SRC) $(GPU_SRC) $(SCHED_SRC) $(MODEL_SRC) $(LOG_SRC) $(BENCH_SRC) src/graph/csr.cpp src/graph/loader.cpp $(REF_SRC) -o $@

clean:
	rm -f tests/test_log tests/test_model tests/test_sched tests/test_deque tests/test_deque_tsan tests/test_sched_debug bin/calibrate_bench tests/test_bench bin/bench_runner

.PHONY: all clean
