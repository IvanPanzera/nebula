# Benchmark record

[Project overview](../README.md) · [Machine-readable evidence](evidence/benchmark-8k.json)

This page separates the historical 8K measurements from the configuration included in the September 9 source snapshot. Exporting the repository did not rerun the large model.

## Historical suite: September 8, 2026

RTX 4070 Ti 12 GB, 96 GB host RAM, Ubuntu/WSL, one text sequence, greedy decoding, Thinking Off, context capacity 8,192, 24 cached experts per layer and prefill chunks of 2,048. The target used the converted Q4 core and the original mixed-quantization routed experts.

Six prompts: arithmetic, grounding, Italian explanation, JSON, Python and reasoning. The aggregate for each mode contains **260 committed decode tokens**. Prompts differ in output length; throughput is weighted by token count through the ratio of total tokens to total decode time.

| Mode | Decode seconds | Committed tokens/s | Draft acceptance |
|---|---:|---:|---:|
| Causal N=0 | 65.533 | 3.967 | Not applicable |
| MTP N=1 | 60.670 | 4.285 | 84.25% |
| MTP N=2 | 61.966 | 4.196 | 76.89% |
| MTP N=3 | 70.441 | 3.691 | 65.44% |
| MTP N=4 fixed | 85.730 | 3.033 | 55.52% |
| Historical adaptive up to N=4 | 60.722 | 4.282 | 79.23% |

N=1 improved the aggregate by approximately 8.0% over causal decoding in this suite. A single Python case reached approximately 5.35 tokens/s with fixed N=4; that case does not describe general throughput.

### Timing protocol

Decode counts committed tokens and excludes the first token predicted during prefill. Loading and prompt processing are separate. The causal reference uses warmed baseline runs bracketing the variants. KV state is reset between trials; the expert cache is retained. Prefill and decode transfer counters are kept separately.

The first model load took 840.886 seconds from the measured HDD. Host memory and PCIe traffic are central to this result; the 12 GB VRAM budget does not describe total system memory consumption. Expert handoff host time and DMA can overlap and should not be added as independent costs.

### Quality observations

The original JSON case is a format failure: correct data was wrapped in Markdown. It remains failed in every mode in the exported summary. A later, separately prompted strict-JSON case passed and does not replace that result.

Local manual review recorded successful arithmetic, grounding and short reasoning. The Python function was identical across the compared modes and passed 105 functional cases in a separate review. The Italian answer was readable but included an overly absolute statement about cloud synchronization and backup. These are small checks, not a broad model-quality evaluation or proof of BF16 equivalence.

## Separate long-context and memory observations

The original 8,096-token retrieval prompt recovered the requested code in all tested modes. A subsequent CLI check reused 8,104 of 8,139 prompt tokens, processed only 35 new tokens and recorded approximately 3.29 seconds before the response. This illustrates prefix reuse on that test, not general first-token latency.

The later 24K development session completed a 24,480-token prefill in approximately 519 seconds inferred from sampled phases. The response was manually interrupted. Retrieval and reuse at 24K remain unverified by that run.

Recorded process RSS was approximately 72–73 GiB. In the monitored 24K session, sampled device free VRAM remained at least 645 MiB, including the observed desktop and driver state; peak process RSS was 72.912 GiB. These are sampled observations, not hard peak bounds. WSL swap was not observed in that run; whole-host Windows paging is a different measurement.

These long-context, manual-quality and memory observations are transcribed from the development report. Their full raw logs are not included in this compact release. The accompanying public JSON contains the six-prompt historical selection metrics only.

## Current configuration and open measurements

The source snapshot uses a 24,576-token context and adaptive MTP N=4–16. On complete acceptance, the progression starts `4 → 4 → 5 → 6 → 8 → 11 → 16`; partial rejections reduce N, and rejection of the first proposal returns it to four. The actual number proposed can be smaller at EOS or context/output limits. Expert transfers resolved by the backend are not token rejections.

Controller tests and CUDA component checks from development are evidence about those components. They do not supply a current full-model speed result. Before claiming completion:

1. Compare N=0, fixed N=1 and adaptive N=4–16 on identical saved prompts and token limits.
2. Include prose, coding, reasoning, strict structured output and long retrieval, keeping failures visible.
3. Record complete loading, prefill, decode and prefix-reuse phases with direct free/reserved VRAM and host memory measurements.
4. Validate output against the same quantized causal target and separately assess language quality.
5. Repeat the setup on an independent machine.

## Evidence provenance

[benchmark-8k.json](evidence/benchmark-8k.json) is a selected-field export of the local `selection_summary.json`. Numeric values are unchanged. It includes the original file's SHA-256, benchmark configuration, aggregate/per-case metrics and quality flags. It omits unrelated environment details and is not the complete raw run archive.

[source-snapshot.json](source-snapshot.json) records the copied source files and hashes. These hashes identify the current source candidate. They are **not** presented as the revision that generated the historical measurements.
