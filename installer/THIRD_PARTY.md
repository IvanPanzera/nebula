Nebula installer dependencies
============================

Model: Qwen/Qwen3.8-Flash-Next; pinned GGUF export by unsloth, revision
38bb39ee97821de2c9009abb7e93950eec396e66. Model files are downloaded at installation,
not embedded in this setup executable. Original tensor formats and Nebula's
approved conversions are recorded in installer/model_recipe.json.

Tokenizer and chat template: Qwen/Qwen3.8-Flash-Next, revision
de4b8e4d43b917e7706784d8bb445c9af86a3540, Qwen Community License 1.0.
The full original license is included in QWEN-LICENSE.txt. Its terms apply to
the model, tokenizer and their derivatives separately from Nebula's engine code.
https://huggingface.co/Qwen/Qwen3.8-Flash-Next

Quantization: GGML from danielhanchen/llama.cpp, commit
d1a92352cbd417fd840b4e765c0b82f5fe3d1d89, MIT.
The pinned source archive, including LICENSE, is downloaded and retained in
the installation cache. Nebula uses ggml-base for offline conversion only.
https://github.com/danielhanchen/llama.cpp

Environment: Ubuntu 24.04.5 WSL image from Canonical; the image includes its
package copyright and license notices under /usr/share/doc.
https://releases.ubuntu.com/noble/

CUDA: NVIDIA CUDA 12.8 compiler and runtime development packages, installed
from NVIDIA's signed WSL Ubuntu repository. NVIDIA's Windows driver remains
the GPU driver. https://docs.nvidia.com/cuda/wsl-user-guide/index.html

Python dependencies and their exact versions are listed in qwen/requirements.txt;
their distributions include their respective license files.
