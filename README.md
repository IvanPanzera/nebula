# Nebula

[Download Windows installer](https://github.com/IvanPanzera/nebula/releases/download/v0.1.0-rc.1/NebulaSetup.exe) · [Installation](#installation) · [Benchmarks](#benchmark-results) · [Contributing](CONTRIBUTING.md)

## Introduction

A nebula grows from a [Dwarf Star](https://github.com/antirez/ds4). Nebula takes another step toward making large language models practical on personal computers.

Nebula is a native C/CUDA inference engine built around **Qwen3.8-Flash-Next**, a mixture-of-experts (MoE) model. It combines NVIDIA GPU computation with CPU execution of expert layers and native multi-token prediction. The reference machine has an **RTX 4070 Ti with 12 GB VRAM, 128 GB DDR4-3600 RAM and a 14-core Intel Core i9-9940X**. The final benchmark campaign measured the following ranges:

| Prefill batch (tokens) | Context capacity (tokens) | Prefill speed | Generation speed |
| --- | --- | --- | --- |
| 2048 | 24576 | 8.43–22.21 token/s | 3.46–14.06 token/s |

These are the minimum and maximum per-response average speeds across the 158 measured responses, described in [Benchmark results](#benchmark-results). The results support batch work and interactive chat, with responsiveness depending on the request.

The aim is to bring this experience to consumer workstations and capable laptops. Hardware allocation, quantization and the coordination of CPU and GPU computation are central to that goal. This overview explains the architecture; the source code provides the implementation details.

## Logo

<p align="center">
  <img src="logo.svg" alt="Nebula logo with three small monoliths and spreading flames" width="300">
</p>

*To bring Promethean fire to as many apes as possible, three small monoliths may work better than one large monolith. The flames inevitably lose some of their intensity as they spread.*

## Architecture

[DwarfStar (ds4)](https://github.com/antirez/ds4) provides the technical and methodological starting point. Nebula specializes that approach for Qwen3.8-Flash-Next through the following changes:

- **Native C and CUDA execution.** Dedicated GPU routines implement the model's attention mechanisms, gated residual connections, MoE layers and native multi-token prediction (MTP).
- **Speculative decoding.** The MTP component acts as a fast *draft*, proposing a sequence of tokens. The main model, called the *target*, verifies that sequence and supplies a correction when a proposal is rejected. Accepted work and the intermediate state needed to resume from the accepted prefix are retained.
- **CPU and GPU coordination.** The draft and the target's attention, shared experts and output head run on the GPU. When a target layer needs significant experts outside its fixed GPU set, the CPU computes that layer's routed MoE and returns the resulting activation vector to the GPU. This event is called an *expert handoff*.
- **Adaptive draft length.** The draft window responds to accepted and rejected proposals, then uses measured execution times to choose a length that improves useful tokens per second.
- **Prioritized GPU resources.** Selected quantization formats and an expert importance ranking, or *hotlist*, determine the GPU-resident components. CUDA Graph reuses recorded sequences of GPU operations to reduce repeated launch overhead.
- **CPU optimization.** On the reference machine, a persistent group of 14 CPU workers shares the work on small token batches. Specialized routines process the 4-bit expert matrices, reuse weights across several tokens and reduce intermediate copies. AVX-512 and AVX2 use CPU instructions that perform arithmetic on several values at once; the engine selects a supported path at startup.
- **A chat WebUI.** The English interface supports multiple saved conversations, renaming and deletion, thinking on/off, context and prefill settings, verification levels, Markdown export and text attachments. Response statistics are shown through the **Statistics** switch.
- **SSD streaming.** Lower-ranked experts and the large n-gram lookup table can stay on SSD. The installation profile chooses their placement according to available RAM and VRAM.

### How a response is generated

The target first reads the prompt during *prefill* and produces the first response token. MTP then proposes a chain of **N** tokens. The target evaluates the chain and checks each proposal in order, using the accepted prefix as its context.

Each of the target's 48 layers contains 512 distinct experts. Its router selects ten for a token and assigns a weight to each. The GPU holds a fixed set of **X experts per layer**, selected from the hotlist. Expert IDs are specific to a layer: expert 121 in layer 12 and expert 121 in layer 30 are separate sets of weights.

There are two token-verification outcomes:

1. **The complete chain is accepted.** Nebula emits the accepted tokens and a bonus next token selected by the target. That token becomes the starting point for the next MTP chain.
2. **A proposal is rejected.** Nebula keeps the preceding accepted tokens and emits the target's correction, selected from scores already calculated during verification. It restores the required intermediate state and realigns MTP with this accepted prefix. Target projections, routing, expert computation and output scores for the retained prefix are reused.

An expert handoff determines where a layer is evaluated. When the missing experts together carry **less than 10% of the router's weight on the current token**, their contribution is omitted. Otherwise, the CPU evaluates the required routed MoE for that layer and token batch, and the GPU continues the target. The final acceptance rule still uses the target's token scores. The GPU retains its fixed experts throughout generation; handoffs transfer activation vectors instead of uploading expert weights.

### Model components and quantization

The model has 125 billion language-model parameters, approximately 6 billion activated per token, a further 51 billion parameters in n-gram embeddings and a 4-billion-parameter MTP component. Draft and target share a vocabulary of 248,320 tokens.

The following weight allocations describe the reference configuration: **27 experts per target layer in VRAM** and **all 512 per layer in RAM**, including CPU copies of the GPU-resident experts. `Q4_K` and `IQ4_NL` are the two 4-bit formats used for routed-expert matrices. Other components retain the formats listed individually. `F32` and `BF16` store floating-point values at 32 and 16 bits respectively. `GDN` denotes Gated DeltaNet attention; `PLE` labels the n-gram lookup components. Hyper-connections combine the model's residual streams, and the indexer selects positions used by sparse attention.

| Component | Format | GPU GiB | GPU GB |
| --- | --- | --- | --- |
| Target and MTP embeddings | Q4_K | 0.333 | 0.358 |
| Target and MTP output head | Q4_K | 0.333 | 0.358 |
| Target resident experts: gate | Q4_K | 1.112 | 1.194 |
| Target resident experts: up | Q4_K | 1.112 | 1.194 |
| Target resident experts: down | IQ4_NL | 1.112 | 1.194 |
| Target: hyper-connection | F32 + IQ4_NL + Q4_K | 0.351 | 0.377 |
| Target: MoE router | F32 | 0.234 | 0.252 |
| Target: shared-expert gate | F32 | 0.000458 | 0.000492 |
| GDN: convolution, norms and scales | F32 | 0.006 | 0.006 |
| GDN: alpha and beta projections | F32 | 0.033 | 0.035 |
| PLE: convolution and norms | F32 | 0.000267 | 0.000287 |
| Target: attention norms | F32 | 0.000023 | 0.000025 |
| Target: indexer and norms | BF16 + F32 | 0.037 | 0.039 |
| GDN: gate, 36 layers | Q4_K | 0.297 | 0.319 |
| GDN: QKV projection, 36 layers | Q4_K | 0.494 | 0.531 |
| Target: shared experts | IQ4_NL + Q4_K | 0.124 | 0.133 |
| GDN: output projection | Q4_K | 0.297 | 0.319 |
| PLE: key and value projections | Q4_K | 0.017 | 0.018 |
| Target: attention K | Q4_K | 0.008 | 0.009 |
| Target: attention output | Q4_K | 0.099 | 0.106 |
| Target: attention Q | Q4_K | 0.198 | 0.212 |
| Target: attention V | Q4_K | 0.008 | 0.009 |
| MTP: attention K | Q4_K | 0.000687 | 0.000737 |
| MTP: attention norms | F32 | 0.000002 | 0.000002 |
| MTP: attention output | Q4_K | 0.008 | 0.009 |
| MTP: attention Q | Q4_K | 0.016 | 0.018 |
| MTP: attention V | Q6_K | 0.001 | 0.001 |
| MTP: shared experts | IQ4_NL + Q4_K | 0.003 | 0.003 |
| MTP: 512 experts, gate | Q4_K | 0.439 | 0.472 |
| MTP: MoE router | F32 | 0.005 | 0.005 |
| MTP: shared-expert gate | F32 | 0.000010 | 0.000010 |
| MTP: 512 experts, up | Q4_K | 0.439 | 0.472 |
| MTP: hyper-connection | F32 + Q4_K + Q5_0 + Q6_K | 0.012 | 0.013 |
| MTP: indexer and norms | BF16 + F32 | 0.003 | 0.003 |
| MTP: input projection and norms | F32 + Q4_K | 0.007 | 0.007 |
| MTP: 512 experts, down | IQ4_NL | 0.439 | 0.472 |
|  | Total | 7.581 | 8.140 |

Additional VRAM holds the context state, temporary work buffers and CUDA Graph resources. The RAM allocations for the reference configuration are:

| Component | Format | RAM GiB |
| --- | --- | --- |
| Target experts in RAM | Q4_K / IQ4_NL | 63.281 |
| PLE table in RAM | (IQ4_NL) | 26.822 |
|  | Total | 90.103 |
| Main CPU buffers |  | 0.292 |

The installer downloads the pinned source weights from Hugging Face and applies Nebula's quantization recipe. The resulting quantizations are stored in the prepared weight files. The repository contains the engine, recipe and hotlist; model downloads are managed by the installer.

### The hotlist

Following ds4's profiling approach, the expert analysis campaign covered **120 requests in 12 subject areas**, with 65,938 input tokens and 43,791 output tokens. Routing traces are available for 43,764 response tokens. These record which experts each layer selected and the weights assigned by its router.

The complete ranking contains all 512 experts for each of the 48 layers. It sorts by activation count, then by accumulated router weight to break ties, and finally by expert ID. The leading X experts stay in VRAM. The leading H experts stay in RAM, including copies of those X GPU experts; the remaining **512 − H** stay on SSD. Equivalently, when Y denotes RAM-only experts, **H = X + Y**.

### Verification levels

The verifier compares the proposed token with the target's scores. Its *greedy* choice is the token with the highest score. For relaxed verification, **R** is the probability assigned by the target to the draft token divided by the probability of its preferred token, at the same response position.

| Level | Acceptance rule | Use |
|:--:|---|---|
| **3 — Strict** | The draft token must equal the target's greedy choice. | Default for technical work, mathematics, coding and complex reasoning. |
| **2 — Limited tolerance** | A different token may be accepted when it is in the target's top three and R ≥ 0.80. At most one such proposal is accepted per block. | Evaluate the speed and response trade-off on the intended workload. The final comparison is reported below. |
| **1 — Broad tolerance** | A different token may be accepted when it is in the target's top ten and R ≥ 0.20. At most four such proposals are accepted per block. | The earlier development comparison found the quality cost too high for the speed gained. |

### Adaptive draft length

Every response starts with **N = 4**, and the controller keeps the window between **4 and 16 proposals**. A baseline rule increases N after fully accepted chains:

```text
N′ = N + ΔN
Consecutive acceptance increments: 0, 1, 1, 2, 3, 5
```

Rejections reduce the baseline:

```text
N′ = N − ΔN
Consecutive rejection decrements: 2, 4, 6
```

Rejecting the first proposal resets the baseline to N = 4. The controller also records the time spent proposing, verifying, saving and restoring state, and realigning MTP. Once it has observations from eight chains, it can move the proposed window by up to two positions when its measured history predicts better useful-token throughput. A 5% improvement threshold reduces reactions to timing noise. The cost of an expert handoff is included in the target's execution time.

## Benchmark results

The **11–12 September 2026 campaign** compared level 3 with level 2 on the RTX 4070 Ti / i9-9940X / 128 GB reference machine. Both levels used a 24,576-token context, a 2,048-token prefill batch, 27 GPU experts per layer, Q4_K/IQ4_NL routed experts, n-gram embeddings in RAM and 14 CPU workers. CPU MoE handoff, the 10% missing-expert threshold, CUDA Graph and intermediate-state recovery were active. Thinking was off and the MTP window varied from 4 to 16. The weights stayed loaded throughout the campaign.

### Protocol

For speed, we used **NVIDIA GenAI-Perf 0.0.16 with Perf Analyzer 2.60.0**. We ran 18 requests, one at a time, using inputs of 512, 1,024 and 2,048 tokens and outputs of 128 tokens: three repetitions for each input length and verification level.

For instruction following, **IFEval** covered 40 questions selected from 541, spanning 25 constraint types. The prompt-level score requires every constraint in a question to be satisfied; the instruction-level score evaluates each constraint separately. The *strict* evaluation applies the checks directly, while *loose* tolerates certain differences in presentation.

For **LiveBench**, we selected 30 questions from the 25 November 2024 dataset: six each in coding, mathematics, reasoning, language and data analysis, spanning 15 task types. The official graders compare answers against references and execute generated programs. Some tasks award partial credit.

The 70 quality questions were selected before testing and submitted to both levels in alternating order. Each question received one answer per level, capped at 2,048 tokens: **140 graded answers**. The campaign took **3.95 hours**, including preparation and checks.

### Performance

| Input (tokens) | Level | Native prefill (tokens/s) | Native generation (tokens/s) | GenAI generation (tokens/s) | First token (s, GenAI) |
| --- | --- | --- | --- | --- | --- |
| 512 | 3 | 20.97 | 6.84 | 6.89 | 25.11 |
| 512 | 2 | 20.86 | 7.66 | 7.69 | 25.19 |
| 1024 | 2 | 21.18 | 6.87 | 6.87 | 49.01 |
| 1024 | 3 | 21.05 | 6.82 | 6.16 | 49.25 |
| 2048 | 3 | 18.49 | 8.04 | 10.14 | 113.94 |
| 2048 | 2 | 19.18 | 8.22 | 8.23 | 107.45 |

Across the 18 NVIDIA tests, average generation speed measured inside Nebula rises from **7.19 tokens/s at level 3 to 7.54 at level 2**, an increase of **4.93%**.

Each row combines three requests. Native speeds divide the total tokens by the total time of the phase. Prefill covers input processing, the chat template and the first output token; decoding generates the subsequent tokens. The GenAI column reports the tool's average speed while receiving responses. Time to first token is the wait before the response starts arriving.

Nebula and NVIDIA therefore observe different processing and reception intervals. In two requests, recorded reception delays account for the most visible differences between the columns. Both measurements and those requests are retained in the results.

| Level | NVIDIA tests native range (tokens/s) | NVIDIA tests native average (tokens/s) | Quality tests native range (tokens/s) | Quality tests native average (tokens/s) |
| --- | --- | --- | --- | --- |
| 3 | 6.36–8.49 | 7.19 | 3.61–13.70 | 8.87 |
| 2 | 6.71–8.51 | 7.54 | 3.46–14.06 | 9.07 |

In the quality tests, average generation reaches **8.87 tokens/s at level 3** and **9.07 at level 2**. These responses can extend to 2,048 tokens; their content and length affect throughput and draft acceptance. Average time to process the question and finish the response is 71.84 seconds at level 3 and 69.62 at level 2, with average response lengths of 509.9 and 500.2 tokens respectively.

The introduction's ranges cover all 158 measured responses: **8.43–22.21 tokens/s for prefill** and **3.46–14.06 for generation**. Each observation is a per-response average. Memory sampled every 30 seconds reached 11.08 GiB on the GPU and 90.78 GiB of resident RAM in the model process. The expert-weight upload counter remained at zero during the responses.

### Response quality

| Measure on the selected sample | Level 3 | Level 2 |
| --- | --- | --- |
| IFEval: prompt strict | 87.50% | 90.00% |
| IFEval: prompt loose | 90.00% | 90.00% |
| IFEval: instruction strict | 91.67% | 93.33% |
| IFEval: instruction loose | 93.33% | 93.33% |
| LiveBench: Coding (n=6) | 83.33% | 83.33% |
| LiveBench: Data analysis (n=6) | 82.17% | 82.17% |
| LiveBench: Language (n=6) | 33.94% | 35.48% |
| LiveBench: Mathematics (n=6) | 36.55% | 53.21% |
| LiveBench: Reasoning (n=6) | 50.00% | 50.00% |
| LiveBench: five-area average | 57.20% | 60.84% |

Across the 70 questions, level 2 scores higher in three cases and level 3 in one; 66 have the same score. The response text is identical in 18 pairs. The 2,048-token cap was reached by eight level-3 answers and seven level-2 answers, mainly in mathematics and reasoning. These responses are included as produced within that cap.

### Evaluation records

The results describe the configuration above, with one request at a time and thinking off. Identical questions and conditions allow a direct comparison between the levels on this selected sample.

We checked the graders against examples with known results and verified that empty extracted programs receive zero points. Generated programs ran in an isolated environment. [Benchmark documentation](docs/BENCHMARKS.md) and [measurement data](docs/evidence/) retain the selected question IDs, scores, timing measurements, source revisions and aggregation definitions.

Official references: [NVIDIA GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf), [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval), [LiveBench](https://github.com/LiveBench/LiveBench).

## Installation

Nebula runs its C/CUDA engine in a dedicated 64-bit Ubuntu environment under **WSL2 on Windows**. The chat interface opens in a browser.

### Requirements

- Windows 10 build 19045 or later, x86-64 CPU and enabled virtualization.
- NVIDIA GPU with at least **8 GB VRAM** and Windows driver **570.65 or later**.
- At least **32 GB RAM**.
- Approximately **151 GiB free on SSD** during installation. The final prepared weights occupy approximately **94.35 GiB**; temporary preparation space and the execution environment account for the installation peak.
- Internet access during setup. Subsequent chat requests run on the computer.

### Automatic setup

Download **[NebulaSetup.exe](https://github.com/IvanPanzera/nebula/releases/download/v0.1.0-rc.1/NebulaSetup.exe)** and accept the Windows administrator prompt. The installer displays progress and completes the following steps. It requests intervention when a prerequisite needs attention, resources need to be released, an active WSL session must close or Windows needs to restart.

1. Checks Windows, CPU architecture, virtualization and the NVIDIA driver.
2. Detects physical RAM, GPU VRAM, CPU cores and logical processors. If several NVIDIA GPUs are present, it selects the single GPU with the most VRAM.
3. Selects one of the 16 approved hardware profiles. RAM and VRAM are rounded down independently: for example, 20 GB VRAM and 80 GB RAM select the 16 GB / 64 GB profile.

#### GPU allocation

| VRAM tier (GB) | GPU experts per layer | Prefill (tokens) | Context (tokens) | Planned free VRAM (GB) |
| --- | --- | --- | --- | --- |
| 8 | 11 | 1024 | 8192 | 1 |
| 12 | 27 | 2048 | 24576 | 2 |
| 16 | 34 | 4096 | 49152 | 3.5 |
| 24 | 54 | 8182 | 98304 | 5 |

#### RAM and SSD allocation

GPU counts are per layer. The RAM column shows **RAM-only experts + CPU copies of GPU experts**, matching the allocation used during CPU MoE handoff.

| VRAM tier (GB) | RAM tier (GB) | RAM-only + GPU copies per layer | SSD experts per layer | N-gram table | Estimated free RAM (GB) |
| --- | --- | --- | --- | --- | --- |
| 8 | 32 | 141 + 11 | 360 | SSD | 8 |
| 8 | 64 | 400 + 11 | 101 | SSD | 8 |
| 8 | 96 | 501 + 11 | 0 | SSD | 27 |
| 8 | 128 | 501 + 11 | 0 | RAM | 33 |
| 12 | 32 | 124 + 27 | 361 | SSD | 8 |
| 12 | 64 | 383 + 27 | 102 | SSD | 8 |
| 12 | 96 | 485 + 27 | 0 | SSD | 27 |
| 12 | 128 | 485 + 27 | 0 | RAM | 33 |
| 16 | 32 | 114 + 34 | 364 | SSD | 8 |
| 16 | 64 | 372 + 34 | 106 | SSD | 8 |
| 16 | 96 | 478 + 34 | 0 | SSD | 27 |
| 16 | 128 | 478 + 34 | 0 | RAM | 33 |
| 24 | 32 | 90 + 54 | 368 | SSD | 8 |
| 24 | 64 | 348 + 54 | 110 | SSD | 8 |
| 24 | 96 | 458 + 54 | 0 | SSD | 27 |
| 24 | 128 | 458 + 54 | 0 | RAM | 33 |

The reserves are the profile's planned headroom. At a fixed RAM tier, a larger GPU profile also needs larger host buffers; this slightly increases the expert count placed on SSD. The 24 GB profile uses the approved prefill value of **8,182**.

4. Checks free SSD space, RAM and VRAM and reports any missing amount. Temporary use by other applications is handled by asking for resources to be released while keeping the selected hardware tier.
5. Selects an internal NTFS SSD for the installation folder, such as `C:\Nebula`. It prefers the system SSD when capacity permits, otherwise the internal SSD with the most free space, and checks the system drive's own requirements.
6. Enables or verifies WSL2 and Windows virtualization components. If a restart is required, setup resumes at the next Windows sign-in.
7. Sets WSL's memory ceiling with the approved Windows reserve and disables WSL swap so explicit RAM/SSD placement governs the model. It preserves unrelated `.wslconfig` settings and backs up the previous file. Applying a change to shared WSL settings waits for active sessions to close.
8. Downloads and verifies Ubuntu, then creates the dedicated **Nebula** WSL distribution, its Linux account and application directories automatically.
9. Installs Python, a dedicated Python environment, the required libraries, build tools and CUDA components. The WebUI uses Nebula's included Python HTTP server.
10. Builds for the detected GPU and prepares AVX-512, AVX2 and scalar CPU implementations. At startup, the engine selects the supported instruction path and uses the available physical cores, up to 28 workers.
11. Downloads the pinned model revision from Hugging Face, checks file sizes and SHA256 integrity hashes, and prepares Nebula's official quantizations. It processes one weight shard at a time and removes each temporary source after verifying its replacement.
12. Generates the GPU hotlist and applies the chosen expert placement. The top X experts per layer stay in GPU memory; the top H stay in RAM, including copies of X; ranks H+1 through 512 remain on SSD and enter bounded temporary RAM buffers when needed.
13. Applies context and prefill limits. The n-gram table stays on SSD for 32, 64 and 96 GB RAM profiles and in RAM for the 128 GB profile. CLI and WebUI use the same saved settings.
14. Runs CPU/CUDA numerical checks and verifies the tokenizer, hotlist, weight index, formats and component sizes. It checks available memory again before finishing. Full model loading starts with the first chat request.
15. Creates **Nebula** desktop and Start-menu shortcuts and **Stop Nebula** in the Start menu. Setup logs are saved in `%ProgramData%\NebulaSetup`; verified model parts and partial downloads are reused after an interrupted installation.

The installer is distributed as a release candidate. Its completed checks and clean-machine validation plan are documented in [Installer validation](docs/INSTALLER-VALIDATION.md).

## Starting Nebula

Open **Nebula** from the desktop or Start menu. The shortcut starts the server in its dedicated WSL environment and opens `http://localhost:8090`. If the service is already running, it opens the existing interface.

Choose **New chat**, enter a message and send it. The first request loads the selected components into RAM and VRAM while the WebUI shows progress. The loaded model then stays available for later requests and new conversations. Closing the browser tab leaves the service running; open Nebula again to return to the chat.

To stop the service and release its memory, choose **Stop Nebula** in the Start menu. This stops Nebula's dedicated WSL distribution. The next launch reloads the weights, and conversations saved in the browser remain available.

If another application occupies port 8090, close it before starting Nebula. Startup details are recorded in `webui-error.log` in the installation folder. Resource messages state how much RAM or VRAM to release before retrying.

## Future development

- Maintenance and improvements based on reproducible reports containing hardware, revision, configuration and observed behavior.
- CPU/GPU allocation and context tuning on additional hardware configurations.
- Extended measurements with different prefill and context lengths, including 16 GB and 24 GB GPUs.
- Support for AMD, Intel and Apple GPUs.
- Evaluation of larger MoE models as available VRAM increases.
- Fresh Windows installation validation and release hardening.

## Credits and license

Thank you to the [Qwen team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for the model, [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) for the GGUF distribution, and the authors of [llama.cpp and GGML](https://github.com/ggml-org/llama.cpp) for quantization formats, tools and reference implementations.

Nebula uses AI assistance for code writing and review, under human direction and verification. The project is independent and preserves its upstream credits and license notices. Engine code is distributed under the [MIT license](LICENSE); the model and tokenizer retain the [Qwen Community License](licenses/QWEN-LICENSE.txt). [Third-party notices](THIRD_PARTY.md) record the pinned sources and their terms.

Special thanks to [antirez](https://github.com/antirez) for ds4 and his educational work on YouTube, which sparked my interest in LLMs and the challenges of local inference.
