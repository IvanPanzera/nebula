# Contributing to Nebula

Describe the behavior being changed, the hardware used and the checks performed. Keep code comments, user-facing messages and documentation in English. Preserve the upstream model/tokenizer assets and their integrity hashes.

The supported product is Windows with WSL2, x86-64 CPUs and one NVIDIA GPU. Development and native checks run inside its Linux environment. Build CPU instruction variants separately; keep feature detection compatible with baseline x86-64. The production model uses fixed GPU experts, CPU MoE handoff, retained-prefix recovery and native MTP with the three documented verification levels.

## Local checks

From a Linux development environment with Python 3.12 and GCC:

```sh
python3 -m venv qwen/build/venv
qwen/build/venv/bin/python -m pip install -r qwen/requirements.txt
bash tools/check_cpu.sh qwen/build/venv/bin/python
```

This suite checks CPU numerical paths, SSD storage, hardware profiles, the download and preparation protocol, verification and the chat interface with test backends. It uses generated small tensors and fixture responses.

Changes to CUDA, attention, speculative state or routing also require the relevant NVIDIA checks:

```sh
make -C qwen -j4 all build/libqwen_cpu.so build/libqwen_probe.so
make -C qwen test PYTHON=build/venv/bin/python
make -C qwen test-spec-capacity
qwen/build/venv/bin/python -m unittest discover -s qwen -p test_large_profiles.py
```

On Windows, validate the installer plan and build the executable with:

```powershell
powershell -NoProfile -File installer/test_installer.ps1
python installer/build_release.py
```

The build writes `dist/NebulaSetup/NebulaSetup.exe` and its SHA256 manifest. See [installer development notes](installer/README.md) and [validation](docs/INSTALLER-VALIDATION.md).

## Performance changes

Compare the same prompts, verification level, response cap and hardware profile. Record model loading separately from prefill and generation. Preserve output-limited answers and failed attempts, and include token counts, elapsed times, expert handoffs and draft acceptance. State any quantization or routing changes explicitly. Benchmark evidence from the reference configuration is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

Keep full-model processes isolated through the existing model lock. Lightweight checks should leave an active chat session usable. Submit each change with a clear problem statement and reproducible validation commands.
