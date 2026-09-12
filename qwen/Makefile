CC ?= gcc
NVCC ?= nvcc
CUDA_ARCH ?= $(shell python3 hardware.py --cuda-arch)
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -fPIC
NVFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler -fPIC
PYTHON ?= build/venv/bin/python

.PHONY: all test test-handoffs test-spec-capacity test-host-weights test-ssd test-storage-math clean
all: build/libqwen.so build/libqwen_gpu.so

build:
	mkdir -p build

build/qwen.o: qwen.c qwen.h gpu.h cpu.h ssd.h | build
	$(CC) $(CFLAGS) -c qwen.c -o $@

build/ssd.o: ssd.c ssd.h | build
	$(CC) $(CFLAGS) -pthread -c ssd.c -o $@

build/gpu.o: gpu.cu gpu.h | build
	@test -n "$(CUDA_ARCH)" || { echo "Set CUDA_ARCH for the selected NVIDIA GPU." >&2; exit 1; }
	$(NVCC) $(NVFLAGS) -c gpu.cu -o $@

CPU_OBJECTS = build/cpu.o build/cpu_scalar.o build/cpu_avx2.o build/cpu_avx512.o
CPU_BASE = $(CFLAGS) -march=x86-64 -mtune=generic -fopenmp

build/cpu.o: cpu_dispatch.c cpu.h | build
	$(CC) $(CPU_BASE) -c cpu_dispatch.c -o $@

build/cpu_scalar.o: cpu.c cpu.h cpu_variant.h cpu_portable.h | build
	$(CC) $(CPU_BASE) -ffp-contract=off -DQC_PREFIX=qcs_ -c cpu.c -o $@

build/cpu_avx2.o: cpu.c cpu.h cpu_variant.h cpu_portable.h | build
	$(CC) $(CPU_BASE) -mavx2 -mfma -mf16c -DQC_AVX2 -DQC_PREFIX=qc2_ -c cpu.c -o $@

build/cpu_avx512.o: cpu.c cpu.h cpu_variant.h | build
	$(CC) $(CPU_BASE) -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -mf16c -DQC_AVX512 -DQC_PREFIX=qc5_ -c cpu.c -o $@

build/libqwen_cpu.so: $(CPU_OBJECTS)
	$(CC) -shared -fopenmp -pthread $^ -lm -o $@.tmp
	mv $@.tmp $@

build/libqwen.so: build/qwen.o build/gpu.o build/ssd.o $(CPU_OBJECTS)
	$(NVCC) -shared $^ -lgomp -o $@.tmp
	mv $@.tmp $@

build/libqwen_gpu.so: build/gpu.o
	$(NVCC) -shared $^ -o $@

build/qwen_probe.o: qwen.c qwen.h gpu.h cpu.h ssd.h | build
	$(CC) $(CFLAGS) -DQWEN_TEST_HOOKS -c qwen.c -o $@

build/libqwen_probe.so: build/qwen_probe.o build/gpu.o build/ssd.o $(CPU_OBJECTS)
	$(NVCC) -shared $^ -lgomp -o $@

test: all build/libqwen_cpu.so
	$(PYTHON) test_cpu_dispatch.py
	$(PYTHON) test_hardware.py
	$(PYTHON) test_gpu.py
	$(PYTHON) test_decode.py
	$(PYTHON) test_optimizations.py
	$(PYTHON) test_verification.py

build/test_handoffs: test_handoffs.c qwen.c qwen.h gpu.h ssd.c ssd.h | build
	$(CC) -O2 -std=c11 -Wall -Wextra -ffunction-sections -fdata-sections $< ssd.c -pthread -Wl,--gc-sections -lm -o $@

test-handoffs: build/test_handoffs
	./build/test_handoffs

build/test_host_weights: test_host_weights.c qwen.c qwen.h gpu.h ssd.c ssd.h | build
	$(CC) -O2 -std=c11 -Wall -Wextra -ffunction-sections -fdata-sections $< ssd.c -pthread -Wl,--gc-sections -lm -o $@

test-host-weights: build/test_host_weights
	./build/test_host_weights

build/test_ssd: test_ssd.c ssd.c ssd.h | build
	$(CC) -O2 -std=c11 -Wall -Wextra -pthread test_ssd.c ssd.c -o $@

test-ssd: build/test_ssd
	./build/test_ssd
	$(PYTHON) test_storage.py

build/test_storage_math: test_storage_math.c qwen.c qwen.h ssd.c ssd.h $(CPU_OBJECTS) | build
	$(CC) -O2 -std=c11 -Wall -Wextra -ffunction-sections -fdata-sections $< ssd.c $(CPU_OBJECTS) -pthread -fopenmp -Wl,--gc-sections -lm -o $@

test-storage-math: build/test_storage_math
	./build/test_storage_math

build/test_spec_capacity.o: test_spec_capacity.c qwen.c qwen.h gpu.h ssd.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build/test_spec_capacity: build/test_spec_capacity.o build/gpu.o build/ssd.o $(CPU_OBJECTS)
	$(NVCC) $^ -lgomp -o $@

test-spec-capacity: build/test_spec_capacity
	./build/test_spec_capacity

clean:
	rm -f build/qwen.o build/gpu.o build/ssd.o $(CPU_OBJECTS) build/libqwen.so build/libqwen_gpu.so build/libqwen_cpu.so
