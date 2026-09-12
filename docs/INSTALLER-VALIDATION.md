# Nebula Setup — development validation

Status: release candidate. The executable and its embedded payload are built;
a complete unattended installation on a clean Windows machine has not yet
been performed. The executable is not Authenticode-signed.

Verified during development:

- All 16 approved hardware profiles, independent downward rounding of RAM
  and VRAM, memory budgets, expert ranking and storage configuration.
- 89 Windows installer assertions covering hardware selection, missing
  resources, WSL configuration preservation and interrupted-install accounting.
- 57 Python tests covering profiles, CUDA, storage, hardware, WebUI, decoding,
  optimizations and verification; the subsequently extended large-context
  CUDA suite also passes all four tests.
- Real CPU dispatch checks for scalar, AVX2 and AVX-512; native MoE, handoff,
  storage and speculative-workspace checks.
- CUDA attention up to 98,304 tokens, prefill capacity 8,192, and sparse index
  selection, including causal ties. The approved 24 GiB profile keeps the
  user-selected prefill of 8,182.
- 33 sampled quantization blocks match the current official derived weights
  byte for byte. A complete second copy of the model was not quantized.
- Native validation of the current model index and the generated portable
  layout for all 1,256 tensors, without loading all weights.
- Resumable downloads, checksum failures, tensor checkpoints, interrupted
  finalization and rejection of corrupt prepared files.
- Availability of all 19 pinned Python dependencies as Linux/Python 3.12
  compatible wheels; installer script syntax and executable compilation.
- Visual inspection of the installer window without executing installation.

The existing Ubuntu environment and running chat were not replaced. A
read-only hardware survey was performed on the development PC. Simulated
profiles establish configuration correctness, not measured performance on
every supported RAM/GPU combination.

The final model occupies approximately 94.35 GiB. The planned fresh-install
peak is approximately 150.81 GiB including the preparation workspace and
environment reserve, independent of the selected hardware profile.

The original development logs are retained with the project archive.
[Release packaging checks](PACKAGING-CHECKS.md) describe the checks repeated on
the English distribution. `release-manifest.json` and `SHA256SUMS.txt` identify
the executable and embedded source files distributed as release assets.
