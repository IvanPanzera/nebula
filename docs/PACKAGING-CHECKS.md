# Release packaging checks — September 12, 2026

The release was assembled from the confirmed engine and checked in a separate
directory. The running development chat and its loaded weights were left intact.

| Check | Result |
| --- | --- |
| Native C/CUDA/header source identity | All 16 files match the confirmed source byte for byte |
| Windows hardware and installer configuration | 89 assertions passed |
| CPU dispatch | Scalar, AVX2 and AVX-512 passed 198 matrix comparisons and nine MoE comparisons; outputs bit-identical |
| Half encodings | All 65,536 encodings checked for each available CPU backend |
| Native handoff, host memory and SSD suites | Passed |
| RAM/SSD MoE equivalence | Passed with 1, 5, 17, 18 and 33 input tokens |
| Interrupted model preparation | Three tests passed |
| Python profiles, storage, hardware, decoding, verification, WebUI and document protocol | 39 tests passed; two CUDA-only cases excluded from the CPU run |
| Python syntax | Passed |
| Windows executable build | Passed; embedded files have a SHA256 manifest |
| Benchmark data export | 140 saved answers and 70 paired rows; no inference rerun |
| Hardware tables | All 16 rows agree with the approved profiles |

The CPU-only suite is available as `bash tools/check_cpu.sh PYTHON`.
CUDA and full-model checks from the development campaign are described in
[installer validation](INSTALLER-VALIDATION.md). The release packaging check
does not run another full-model benchmark or a clean-machine installation.

The source package contains engine code, the complete expert ranking,
hardware profiles, the pinned tokenizer, installer sources and benchmark
evidence. Model weights are obtained and prepared by the installer. Local
runtime profiles, personal paths, chat history, cached credentials, build outputs
and the earlier light-model integration are excluded.

The standalone setup executable is distributed as a release asset with
`SHA256SUMS.txt` and `release-manifest.json`. Its complete unattended installation
on a clean Windows machine remains a release-candidate validation step.
