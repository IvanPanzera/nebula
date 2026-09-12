# Nebula engine

The current configuration and measured results are described in the [project README](../README.md). Install from [NebulaSetup.exe](https://github.com/IvanPanzera/nebula/releases/download/v0.1.0-rc.1/NebulaSetup.exe), or follow the [developer checks](../CONTRIBUTING.md) when modifying the source.

## Source layout

| File | Responsibility |
|---|---|
| `qwen.c`, `qwen.h` | Model loading, target/MTP execution, recurrent state and expert handoff |
| `gpu.cu`, `gpu.h` | CUDA operators, attention, verification and reusable graphs |
| `cpu.c`, `cpu_dispatch.c`, `cpu_portable.h`, `cpu_variant.h` | Quantized CPU arithmetic and instruction-set selection |
| `ssd.c`, `ssd.h` | Bounded reads for experts kept on SSD |
| `decode.py` | Speculative proposal, verification, correction and draft-window timing policy |
| `native.py` | Python bindings and model/quantization integrity checks |
| `profiles.py`, `hardware.py`, `hardware_profiles.json` | Hardware detection and the 16 approved installation profiles |
| `storage.py`, `storage.json` | Host expert and n-gram storage policy |
| `expert_ranking.json`, `hotlist.json` | Complete per-layer ranking and GPU-resident expert selection |
| `chat.py`, `web_server.py`, `web_worker.py`, `web/` | Command-line chat and the English browser interface |
| `documents.py`, `document_worker.py` | Document boundaries and optional document-conversion integration |

## Reference configuration

The development defaults are 24,576 context tokens, prefill batches of 2,048, 27 fixed GPU experts per target layer and 14 CPU workers. Routed expert matrices use Q4_K for gate/up and IQ4_NL for down. The native MTP proposes 4–16 tokens and shares the target's vocabulary.

When significant experts are missing from GPU memory, the CPU evaluates the routed MoE for that layer and token batch. The GPU continues attention, shared experts and the output head. Missing experts may be omitted when their combined current-token router weight is below 10%. A rejected draft token is corrected from the target's existing scores; retained target work is reused.

## RAM and SSD placement

`storage.json` holds the selected policy. The installer writes it together with `runtime_profile.json` and the matching hotlist. These settings are validated at process startup.

| Setting | Meaning |
|---|---|
| `ram_experts_per_layer` | Total RAM-resident experts per layer, including CPU copies of GPU-resident experts. SSD holds the other `512 - ram_experts_per_layer`. |
| `ngram` | `ram` or `ssd`; the SSD mode uses a bounded lookup cache. |
| `buffer_mib` | Additional temporary RAM budget for cold expert reads. |
| `io_threads` | Persistent concurrent SSD readers, from 1 to 4. |

The expert ranking orders activation count descending, accumulated router weight descending for ties, then expert ID ascending. It contains all 512 experts for every target layer. The most highly ranked experts occupy GPU/RAM; the tail stays on SSD. Expert identity is the pair `(layer, expert ID)`.

Cold weights are read into bounded host buffers when the CPU needs them and released after the corresponding work. The engine subdivides large CPU prefill work to respect the temporary-memory budget. Storage counters expose requested read bytes, read counts and waiting times.

## Building and starting from prepared assets

Inside the installed WSL environment:

```sh
make -C qwen -j4
qwen/build/venv/bin/python qwen/chat.py
qwen/build/venv/bin/python qwen/web_server.py
```

Run one entry point at a time. Native execution requires the prepared `work/qwen/model.index`, matching verification manifest, generated profile and weights. The tokenizer is bundled in `assets/tokenizer/` and checked against its pinned manifest.

The standard Windows installation creates desktop and Start-menu launchers. Optional OCR integration uses separately configured document-conversion assets; plain-text chat is available with the base installation.
