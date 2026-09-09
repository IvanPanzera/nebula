# Source export checks — September 9, 2026

The private repository candidate was checked before upload. These checks do not load Flash-Next weights or establish a new inference speed result.

| Check | Result |
|---|---|
| Python syntax | All 40 Python source files parsed |
| Documentation | Local Markdown file links resolved |
| Historical metrics | All six aggregate token/time ratios match the exported throughput values |
| Speculative controller | 10 tests passed |
| WebUI protocol and worker | 13 tests passed, using reference/simulated backends and the official tokenizer |
| Optional light-model bridge | 10 tests passed with simulated model behavior |
| Document bridge | 5 tests passed without OCR model inference |
| C handoff scheduler | Cold/warm cache, partial misses, batching, per-layer counters, MTP exclusion and reset passed with inert GPU operations |

The WebUI tests initially failed because the clean export had no tokenizer assets. Supplying the four already-verified official tokenizer files and their manifest resolved that prerequisite; no engine code was changed. Those downloaded/generated assets are excluded from Git and obtained through the setup guide.

Checks used the existing development Python environment (Python 3.14.4 on Ubuntu/WSL), except the standalone decoder test, which also passed with Windows Python 3.10. A clean dependency installation, CUDA numerical regression and full-model benchmark were not rerun for this packaging task.

Source hashes are in [source-snapshot.json](source-snapshot.json). The current engine source differs in configuration from the historical 8K benchmark, as explained in [BENCHMARKS.md](BENCHMARKS.md).
