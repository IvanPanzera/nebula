$ErrorActionPreference = 'Stop'
$port = 8090
try {
    try { $status = Invoke-RestMethod "http://127.0.0.1:$port/api/status" -TimeoutSec 2 } catch { $status = $null }
    if ($status -and $status.app -eq 'qwen-webui' -and $status.installation -eq 'nebula-wsl') { Start-Process "http://localhost:$port"; exit 0 }
    if ($status) { throw 'Port 8090 is used by another application. Close it and open Nebula again.' }
    $arguments = @('-d','Nebula','-u','nebula','--','/home/nebula/app/qwen/build/venv/bin/python','-u','/home/nebula/app/qwen/web_server.py','--port',"$port")
    $process = Start-Process -FilePath 'wsl.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'webui.log') -RedirectStandardError (Join-Path $PSScriptRoot 'webui-error.log')
    for ($i=0;$i -lt 90;$i++) {
        Start-Sleep -Seconds 1
        if ($process.HasExited) { throw 'Nebula could not start. See webui-error.log in the installation folder.' }
        try { $status=Invoke-RestMethod "http://127.0.0.1:$port/api/status" -TimeoutSec 1 } catch { continue }
        if ($status.app -eq 'qwen-webui' -and $status.installation -eq 'nebula-wsl') { Start-Process "http://localhost:$port"; exit 0 }
    }
    throw 'Nebula did not become available. See webui-error.log in the installation folder.'
} catch {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show($_.Exception.Message,'Nebula') | Out-Null
    exit 1
}
