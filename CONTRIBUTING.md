# Contributing to City of Brass

Contributions that improve correctness, reduce memory traffic or make results easier to reproduce are welcome. The initial target is one text sequence on a 12 GB NVIDIA GPU with host-memory expert offload.

## Report a result

Include the source revision, GPU and driver, CUDA version, OS/WSL version, system RAM, storage layout and exact model revisions. Record quantization, context capacity, actual prompt length, prefill chunk size, cached experts, draft policy and thinking mode.

Attach the prompt, generation limit, committed-token count, prefill and decode times, acceptance, expert transfers, sampled memory usage and output correctness. Redact private prompts before uploading. Distinguish generated tokens, accepted proposals and discarded proposals.

## Compare changes

Use identical prompts and the same quantized target, with causal N=0 baselines around the candidate runs. Separate loading, prefill, first-token latency and decode; compare committed tokens per second. Preserve failed cases and responses cut off by their token limit.

Do not attribute historical N=1 or adaptive-up-to-4 measurements to the current N=4–16 policy. Do not treat a CUDA allocation check, controller mock or component test as validation of the complete model.

## Development checks

From the repository root on the configured Linux/CUDA environment:

```sh
qwen/build/venv/bin/python qwen/test_decode.py
qwen/build/venv/bin/python qwen/test_webui.py
make -C qwen test-handoffs
make -C qwen test-spec-capacity
make -C qwen test
```

The decoder and WebUI tests use reference or simulated backends. The handoff test uses inert GPU operations. `make test` includes GPU numerical tests; full-model checks need the pinned weights and a free model session. Keep the instance lock and avoid simultaneous copies of the large model.

Changes to routing, quantization, state restore or numerical kernels should include the relevant causal/token or numerical comparison and a before/after performance result. Keep public interfaces small and preserve existing license notices.

See [setup](qwen/README.md) and [benchmark methodology](docs/BENCHMARKS.md).
