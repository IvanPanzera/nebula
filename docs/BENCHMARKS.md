# Standard benchmark campaign — September 11–12, 2026

Nebula was tested with NVIDIA GenAI-Perf, IFEval and LiveBench on an Intel
Core i9-9940X, 128 GB of RAM and a 12 GB NVIDIA GPU under Windows/WSL2.
The configuration used 14 CPU threads, 27 resident experts per layer, a
24,576-token context and a 2,048-token prefill batch. Native MTP, CUDA graphs,
retained-prefix recovery and the time-aware draft controller were enabled.
Thinking was off. Expert handoffs executed the complete routed MoE of the layer
on the CPU; resident experts remained fixed in GPU memory.

## Protocol

- **GenAI-Perf:** 512, 1,024 and 2,048 input tokens; 128 output tokens; three
  repetitions per input length and verification level, concurrency one:
  18 measured requests. Warmup requests were recorded separately.
- **IFEval:** 40 questions, covering 60 checkable instructions, at levels 3 and 2.
- **LiveBench:** 30 questions from release 2024-11-25, with six in each of
  coding, mathematics, reasoning, language and data analysis, at both levels.
- Quality answers had a 2,048-token limit. Each question was run once per level,
  with alternating level order. Deterministic hash ordering and round-robin
  selection across instruction/task groups selected the sample before generation.
- All 70 pairs completed. The measured set comprises 158 requests and 73,013
  output tokens. Generation and preparation finished in approximately 3.95 hours.

Level 3 requires the draft token to match the target's greedy choice. Level 2
also permits a draft token in the target's top three when its probability is
at least 80% of the target's first choice, with at most one such acceptance per
block. All other model settings were identical.

## Results

| Measurement | Level 3 | Level 2 |
| --- | ---: | ---: |
| GenAI-Perf workload: native decode, weighted tokens/s | 7.19 | 7.54 |
| Quality workload: native decode, weighted tokens/s | 8.87 | 9.07 |
| IFEval strict prompt accuracy | 87.50% | 90.00% |
| IFEval loose prompt accuracy | 90.00% | 90.00% |
| IFEval strict instruction accuracy | 91.67% | 93.33% |
| IFEval loose instruction accuracy | 93.33% | 93.33% |
| LiveBench mean across five categories | 57.20 | 60.84 |
| Answers reaching the output limit | 8 | 7 |

Across the 70 paired scores, level 2 scored higher on three questions, level 3
on one, and 66 tied. Eighteen pairs produced identical text. These observations
describe the selected questions and settings; sampling uncertainty is especially
large within the six-question LiveBench categories.

The complete measured set ranged from **8.43 to 22.21 prefill tokens/s** and
**3.46 to 14.06 decode tokens/s**. These are per-request extrema. Native decode
throughput divides the tokens after the first output token by the decode time;
the first token belongs to prefill. Weighted rates divide summed token counts
by summed phase durations. A one-token answer has no decode-rate measurement.

## Timing, resources and grading

The saved summary includes both native phase timings and GenAI-Perf client
statistics. Client timings include adapter and streaming delivery overhead.
Two runs had appreciable delivery delays: one 2,048-token level-3 request had
about 7.52 seconds of additional delay before the first text, and one
1,024-token level-3 request had a client generation span about 6.85 seconds
longer than the native span. Both remain in the reported measurements.

The resource monitor recorded 406 samples. Peak total GPU memory use was
11.08 GiB and peak worker resident memory was 90.78 GiB; measured WSL swap use
remained zero. The expert-upload counter remained zero. An SSD-byte counter
was not present in these responses and is retained as null in the export.

IFEval used the upstream deterministic checks. LiveBench used the pinned,
release-matched upstream evaluator. Two HTML grading attempts initially lacked
the `lxml` dependency; their original saved answers were graded after restoring
the dependency. No response was regenerated for that recovery. Grader controls
passed and hashes of the 13 monitored engine files were checked.

## Evidence and provenance

- [Protocol](evidence/protocol.json): selection seed, ordering, limits and metrics.
- [Full aggregates](evidence/summary.json): category scores, timings, resource
  statistics and paired comparisons at their recorded precision.
- [Paired results](evidence/paired-results.csv): one row per question.
- [Saved answers](evidence/answers.jsonl): all 140 original answers, recorded
  scores and scalar metrics, plus the SHA256 of each source result file.
- [Dataset revisions](evidence/dataset-sources.json),
  [upstream source checksums](evidence/upstream-sources.json) and
  [LiveBench grading revision](evidence/grader-revision.json).
- [Export manifest](evidence/export-manifest.json): hashes of the exported data.

The exported files were produced from the completed campaign. Dataset questions
can be retrieved from the pinned upstream sources using the IDs in the paired
results. Original benchmark data and evaluator code retain their upstream
licenses. The selected benchmark questions were public test data; no user chat
history is included in this export.

Upstream projects: [NVIDIA GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf),
[IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval),
and [LiveBench](https://github.com/LiveBench/LiveBench).
