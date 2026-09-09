CC ?= gcc
NVCC ?= nvcc
CUDA_ARCH ?= sm_89
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -fPIC
NVFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler -fPIC
PYTHON ?= build/venv/bin/python

.PHONY: all test test-handoffs test-spec-capacity clean
all: build/libqwen.so build/libqwen_gpu.so

build:
	mkdir -p build

build/qwen.o: qwen.c qwen.h gpu.h | build
	$(CC) $(CFLAGS) -c qwen.c -o $@

build/gpu.o: gpu.cu gpu.h | build
	$(NVCC) $(NVFLAGS) -c gpu.cu -o $@

build/libqwen.so: build/qwen.o build/gpu.o
	$(NVCC) -shared $^ -o $@.tmp
	mv $@.tmp $@

build/libqwen_gpu.so: build/gpu.o
	$(NVCC) -shared $^ -o $@

build/qwen_probe.o: qwen.c qwen.h gpu.h | build
	$(CC) $(CFLAGS) -DQWEN_TEST_HOOKS -c qwen.c -o $@

build/libqwen_probe.so: build/qwen_probe.o build/gpu.o
	$(NVCC) -shared $^ -o $@

test: all
	$(PYTHON) test_gpu.py
	$(PYTHON) test_decode.py

build/test_handoffs: test_handoffs.c qwen.c qwen.h gpu.h | build
	$(CC) -O2 -std=c11 -Wall -Wextra -ffunction-sections -fdata-sections $< -Wl,--gc-sections -lm -o $@

test-handoffs: build/test_handoffs
	./build/test_handoffs

build/test_spec_capacity.o: test_spec_capacity.c qwen.c qwen.h gpu.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build/test_spec_capacity: build/test_spec_capacity.o build/gpu.o
	$(NVCC) $^ -o $@

test-spec-capacity: build/test_spec_capacity
	./build/test_spec_capacity

clean:
	rm -f build/qwen.o build/gpu.o build/libqwen.so build/libqwen_gpu.so
