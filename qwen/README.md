# Preparing the Qwen engine

[Project overview](../README.md) · [Historical benchmarks](../docs/BENCHMARKS.md)

This guide describes the source snapshot's Flash-Next CLI and WebUI. It is a research setup, with a substantial model download and offline conversion step. The preparation sequence comes from the development machine; this exported repository still needs a clean installation on a second machine.

## Tested hardware and software assumptions

| Component | Setup |
|---|---|
| GPU | RTX 4070 Ti, 12 GB VRAM; the Makefile defaults to CUDA `sm_89` |
| RAM | 96 GB installed; approximately 72–73 GiB process RSS in recorded runs |
| OS | Windows with Ubuntu/WSL; the engine uses Linux interfaces including `mmap` and `fcntl` |
| Python | 3.14 for the preparation tools; MTP conversion uses `Executor.map(buffersize=...)` |
| Build tools | GCC, Make and NVIDIA CUDA with `nvcc`; reference tools install pinned CMake/Ninja |
| Storage | Original GGUF files approximately 113 GB combined; additional room for derived weights, the roughly 26.8 GiB PLE copy, build files and free-space reserve |

On the measured machine, GGUFs were stored on a SATA HDD and the n-gram/PLE cache was on an NVMe-backed WSL filesystem. Cold model loading took about 14 minutes. This is a storage-specific observation.

## 1. Python environment

Run the Linux commands from the repository root. Python 3.14, its venv support and a working CUDA toolchain must already be installed:

```sh
python3.14 -m venv qwen/build/venv
qwen/build/venv/bin/python -m pip install -r qwen/requirements.txt requests
```

`requests` is required by the asset and reference preparation scripts and is supplied explicitly here. The existing `bootstrap.py` is retained in the snapshot, but the commands above make that prerequisite explicit.

## 2. Download the pinned assets

```sh
qwen/build/venv/bin/python qwen/tokenizer_assets.py
qwen/build/venv/bin/python qwen/assets.py --download
```

The scripts audit metadata and verify downloaded files. They populate `models/qwen/`, `work/qwen/` and the tokenizer directory under `work/research/`. These are generated local directories and are ignored by Git.

| Asset | Pinned repository and revision |
|---|---|
| Target and tokenizer | `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Target GGUF and MTP | `unsloth/Qwen3.8-Flash-Next-GGUF` at `38bb39ee97821de2c9009abb7e93950eec396e66` |
| Offline GGML quantizer / comparison engine | `danielhanchen/llama.cpp` at `d1a92352cbd417fd840b4e765c0b82f5fe3d1d89` |

The target is the four-shard `UD-Q4_K_XL` distribution, approximately 111.33 GB. Its shared MTP GGUF adds approximately 1.907 GB. The native loader is specific to these assets.

## 3. Prepare quantization and the SSD PLE cache

```sh
qwen/build/venv/bin/python qwen/reference_engine.py
qwen/build/venv/bin/python qwen/quantize_mtp.py
qwen/build/venv/bin/python qwen/quantize_core.py
```

The pinned reference build supplies the offline GGML quantizer. Flash-Next production inference uses the native engine in this repository. The target core uses Q4_K or IQ4_NL, while routers and normalizations retain their source precision. Routed experts retain the original mixed Q4_K/Q5_1/Q5_K/Q8_0 formats. MTP uses its shared Q4_K_M asset and a derived IQ4_NL down projection.

Copy the unchanged PLE tensor to a fast Linux filesystem. On native Linux:

```sh
qwen/build/venv/bin/python qwen/cache_ple.py --directory ~/.cache/city-of-brass/qwen
```

On WSL the copier also requires the physical free space of the drive containing the Linux virtual disk. For a virtual disk on **C:**, run from PowerShell in the repository root:

```powershell
$taskFreeBytes = [System.IO.DriveInfo]::new('C:').AvailableFreeSpace
wsl -d Ubuntu -- qwen/build/venv/bin/python qwen/cache_ple.py --host-free-bytes $taskFreeBytes
```

Select the actual backing drive if it differs. The copier checks space for the PLE data plus 24 GiB headroom and verifies the copy. WSL virtual free space alone does not establish available physical disk space.

## 4. Create the index and build

With all model processes closed:

```sh
qwen/build/venv/bin/python qwen/model_index.py --mtp-iq4 --core-q4
make -C qwen -j2
```

The default build architecture is `sm_89`. Changing it for another CUDA GPU requires separate numerical and memory validation; the 12 GB result was measured on the RTX 4070 Ti.

## 5. Run locally

```sh
qwen/build/venv/bin/python qwen/chat.py
qwen/build/venv/bin/python qwen/chat.py --prompt "Explain how a mixture-of-experts router works."
qwen/build/venv/bin/python qwen/chat.py --draft-max 1 --fixed-draft
qwen/build/venv/bin/python qwen/chat.py --draft-max 0
```

In interactive mode, `/reset` clears the conversation and keeps weights loaded; `/exit` closes the model. Default context is 24,576 tokens, including prompt and response; default response limit is 2,048. Greedy decoding and Thinking Off are the defaults. The speculative policy starts at four proposals and can rise to sixteen.

For the browser interface:

```sh
qwen/build/venv/bin/python qwen/web_server.py
```

Open `http://localhost:8090`. The server binds to loopback and processes one model session. Close the CLI session before using the WebUI. The first message loads the weights; subsequent messages can reuse the model and conversation prefix. Stop may wait for the current computation to finish.

The source includes optional light-model and OCR bridges. Their model configurations, OCR runtime and weights are not installed by this guide. The Windows WebUI launcher from the development workspace points to a separate local installation and is intentionally excluded; use the command above. The included `avvia_qwen.bat` is the repository-relative CLI launcher.

## Verification and benchmark commands

```sh
qwen/build/venv/bin/python qwen/test_decode.py
qwen/build/venv/bin/python qwen/test_webui.py
make -C qwen test-handoffs
make -C qwen test-spec-capacity
make -C qwen test

qwen/build/venv/bin/python qwen/benchmark.py --include-adaptive --long-context --multiturn --out work/qwen/new-benchmark
qwen/build/venv/bin/python qwen/analyze_benchmark.py work/qwen/new-benchmark/report.json --out work/qwen/new-benchmark/summary.json
```

Full benchmarks occupy the GPU and load the large model. The current benchmark's adaptive mode uses N=4–16 and must produce its own result. `check_context.py` and `check_web_context.py` are retained development helpers that expect additional historical run files under `work/qwen/benchmark_daily_q4/`; the compact public evidence JSON alone is not their complete fixture.

The source snapshot and unit checks establish packaging and controller behavior. They do not replace full-model numerical, quality or throughput validation.
