$ErrorActionPreference = 'Stop'
try {
    # Only Nebula's dedicated environment is stopped. Other WSL distributions
    # are unaffected; the next launch recreates the server and loads the model.
    & wsl.exe --terminate Nebula
    if ($LASTEXITCODE -ne 0) { throw 'Nebula could not be stopped. Check that its WSL environment is available.' }
} catch {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show($_.Exception.Message,'Nebula') | Out-Null
    exit 1
}
