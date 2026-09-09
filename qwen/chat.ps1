# Launch the local WSL engine from any PowerShell working directory.
$ErrorActionPreference = 'Stop'
$taskQwenArguments = @($args)
# Resolve file arguments from the caller's directory, before switching to the
# project. Absolute Linux paths already address the WSL filesystem directly.
for ($taskArgumentIndex = 0; $taskArgumentIndex -lt $taskQwenArguments.Count - 1; $taskArgumentIndex++) {
    if ($taskQwenArguments[$taskArgumentIndex] -notin @('--prompt-file', '--report')) { continue }
    $taskArgumentIndex++
    $taskFilePath = [string]$taskQwenArguments[$taskArgumentIndex]
    if ($taskFilePath.StartsWith('/')) { continue }
    if (-not [System.IO.Path]::IsPathRooted($taskFilePath)) {
        $taskFilePath = Join-Path -Path (Get-Location).ProviderPath -ChildPath $taskFilePath
    }
    $taskFilePath = [System.IO.Path]::GetFullPath($taskFilePath)
    $taskLinuxPath = & wsl.exe -d Ubuntu -- wslpath -a -u $taskFilePath
    if ($LASTEXITCODE -ne 0) { throw "Cannot translate the file path to WSL: $taskFilePath" }
    $taskQwenArguments[$taskArgumentIndex] = $taskLinuxPath
}
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    & wsl.exe -d Ubuntu -- qwen/build/venv/bin/python -u qwen/chat.py @taskQwenArguments
    $taskQwenExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
if ($taskQwenExitCode -ne 0) {
    throw "Qwen terminated with exit code $taskQwenExitCode."
}
