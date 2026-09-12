# Nebula Setup

Run **NebulaSetup.exe** and accept the Windows administrator prompt. Setup
detects hardware, selects the approved profile, installs a dedicated **Nebula**
WSL2 distribution, builds the engine, downloads and verifies the model, prepares
the official quantizations and creates desktop and Start-menu shortcuts.

No Python, terminal commands, Linux account creation or model configuration is
required from the user. Setup displays progress and saves a detailed log in
`%ProgramData%\NebulaSetup`. If Windows needs a restart, Setup resumes after the
next sign-in. If a resource or prerequisite is missing, the window gives the
required action; run Setup again afterwards. Completed model parts and interrupted
downloads are reused.

Open **Nebula** from the desktop when installation finishes. The model loads
with the first message. The launcher does not require administrator rights.
Closing the browser leaves the model available for another conversation. Use
**Stop Nebula** in the Start menu to stop Nebula's dedicated WSL environment and
release its memory. Other WSL distributions are unaffected. The next launch
loads the model again; saved browser chats remain available.

## Hardware selection

Windows x86-64, WSL2 and an NVIDIA GPU are required. The installer uses the 16
rows actually present in the approved document: VRAM 8/12/16/24 GiB crossed with
RAM 32/64/96/128 GiB. Each dimension is rounded down independently: 20 GiB VRAM
uses the 16 GiB profile; 80 GiB RAM uses the 64 GiB profile. Extra memory does not
silently increase expert residency. Multiple GPUs are not added together.

The small allowance for NVIDIA's reserved memory recognizes nominal card sizes
(for example, a 12 GiB card reported as 12,282 MiB).

Per layer, the first X experts in the complete campaign ranking stay in GPU;
the first H experts stay in RAM, including the copies of those X experts.
Ranks H+1 through 512 remain on SSD and are read into a bounded temporary buffer
only when needed. The N-gram table stays on SSD for the 32/64/96 GiB RAM tiers
and in RAM for the 128 GiB tier. No model weights are uploaded on expert handoff:
the CPU executes the layer's complete MoE under the official handoff policy.

The document's 24 GiB GPU row specifies a prefill of **8182**, which is preserved
exactly. Contexts are 8192, 24576, 49152 and 98304 tokens for the four GPU tiers.

## Storage and prerequisites

Setup chooses an internal NTFS SSD. It prefers the system SSD when it has
enough space; otherwise it selects the internal SSD with most free space.
The install directory is `DRIVE:\Nebula`. Existing unrelated directories and
WSL distributions are not overwritten.

The final repacked weights occupy approximately **94.35 GiB**. Installation
requires approximately **150.81 GiB free** at its peak, including the temporary
source shard, WSL, CUDA/build dependencies and working reserve. Setup reports
the actual missing amount and also checks free RAM and VRAM. It retains the
selected hardware tier when another application occupies memory and asks for
that memory to be released.

The NVIDIA Windows driver must be version 570.65 or newer and virtualization
must be enabled. Missing prerequisites produce instructions instead of a
silent failure. Linux GPU drivers are not installed inside WSL.

WSL's RAM ceiling is physical RAM minus the approved Windows reserve. Existing
unrelated `.wslconfig` entries are preserved and the old file is backed up.
Because that file is shared by WSL distributions, Setup waits for running WSL
applications to be closed before applying a change; it does not terminate them.
WSL swap is disabled to keep the explicit RAM/SSD policy from becoming implicit
paging. CPU execution selects AVX-512, AVX2 or scalar kernels at runtime and uses
up to 28 physical cores available to WSL.

## Verification and release scope

Setup checks CPU dispatch, CUDA equations (including 96K attention), hardware
profiles, tokenizer and model provenance before completing. The first chat
request performs the full model allocation. Resource tiers describe memory
allocations, not equal generation speed across different hardware.

This installer is a release candidate. The development validation includes
simulated profiles, a read-only survey of the development PC, real CPU/CUDA
tests and sampled equivalence of the quantization recipe. A complete unattended
installation on a clean Windows machine is still a release validation step.
The executable is not yet Authenticode-signed.

## Build and inspect

Source is in `installer/`; runtime profiles are in `qwen/hardware_profiles.json`.
The installer has an embedded payload protected by a SHA256 manifest.

For a read-only hardware plan:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File installer/setup.ps1 -PlanOnly
```

The audited tensor layout and memory budgets are committed as
`model_recipe.json` and `budgets.json`. To validate and build from the repository
root on Windows:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File installer/test_installer.ps1
python installer/build_release.py
```

The build uses the Windows .NET Framework C# compiler and writes the executable,
checksums and payload manifest to `dist/NebulaSetup/`. Python 3.12 or newer is
required for this developer build. Users run the resulting executable directly.
