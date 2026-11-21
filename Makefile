# T4 (Turing, sm_75) is the only Kaggle GPU since the P100 retirement.
NVCC ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CXX ?= g++
ARCH ?= -gencode arch=compute_75,code=sm_75
NVFLAGS = -O3 -std=c++17 -fmad=false -lineinfo $(ARCH) -Xptxas -v
CXXFLAGS = -O2 -std=c++17 -ffp-contract=off -Wall -Wextra

all: bin/mdms_twin bin/mdms_gpu

bin/mdms_twin: src/mdms_twin.cpp src/mdms_core.cuh src/host_common.hpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ src/mdms_twin.cpp

bin/mdms_gpu: src/mdms_gpu.cu src/mdms_core.cuh src/host_common.hpp
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -o $@ src/mdms_gpu.cu

twin: bin/mdms_twin

clean:
	rm -rf bin
.PHONY: all twin clean
