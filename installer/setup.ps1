param([switch]$PlanOnly, [string]$InventoryFile = '', [string]$SetupExecutable = '')
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'hardware.ps1')
$payload = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$data = Join-Path $env:ProgramData 'NebulaSetup'

function Log([string]$Message) { Write-Output $Message }
function Native([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program $($Arguments -join ' '). See the installation log, resolve the reported error and run Setup again." }
}
function Schedule-Resume {
    if (!$SetupExecutable -or !(Test-Path -LiteralPath $SetupExecutable)) { throw 'A Windows restart is required. Restart Windows and run Nebula Setup again.' }
    New-Item -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce' -Force | Out-Null
    New-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce' -Name 'NebulaSetup' -PropertyType String -Value ('"' + $SetupExecutable + '" --resume') -Force | Out-Null
    Log 'Windows needs a restart. Restart when convenient; Nebula Setup will resume automatically after sign-in.'
    exit 3010
}

try {
    $hardware = if ($InventoryFile) { Read-Json $InventoryFile } else { Get-WindowsHardware }
    $existing = ''
    $statePath = Join-Path $data 'setup-state.json'
    if (Test-Path -LiteralPath $statePath) { $existing = (Read-Json $statePath).InstallRoot }
    $plan = Get-InstallPlan $hardware $payload $existing
    if ($PlanOnly) { $plan | ConvertTo-Json -Depth 20; exit 0 }
    if ($InventoryFile) { throw 'Simulated hardware is allowed only with -PlanOnly.' }
    $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Run Nebula Setup as administrator.' }
    if (!$plan.CanInstall) { throw ($plan.RequiredActions -join "`n") }
    Log ('Hardware profile: {0} GiB VRAM / {1} GiB RAM; {2} GPU experts, {3} RAM experts, {4} SSD experts per layer.' -f $plan.Profile.vram_gib, $plan.Profile.ram_gib, $plan.Profile.gpu_experts, $plan.Profile.ram_experts, $plan.Profile.ssd_experts)
    Log ('Context {0}; prefill {1}; N-gram {2}.' -f $plan.Profile.context, $plan.Profile.prefill, $plan.Profile.ngram)
    New-Item -ItemType Directory -Path $data -Force | Out-Null
    Write-Json $statePath $plan
    $install = $plan.InstallRoot
    New-Item -ItemType Directory -Path $install -Force | Out-Null
    foreach ($path in @($install, (Join-Path $install 'model'), (Join-Path $install 'cache'), (Join-Path $install 'wsl'))) {
        if ((Test-Path -LiteralPath $path) -and ((Get-Item -LiteralPath $path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "The installation path $path is a filesystem link. Move that link, then run Setup again."
        }
    }
    $acl = Get-Acl -LiteralPath $install
    $userSid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    $acl.SetAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($userSid,'Modify','ContainerInherit,ObjectInherit','None','Allow')))
    Set-Acl -LiteralPath $install -AclObject $acl
    $markerPath = Join-Path $install '.nebula-install.json'
    $wasComplete = $false
    if (Test-Path -LiteralPath $markerPath) { $m = Read-Json $markerPath; $wasComplete = ($m.PSObject.Properties.Name -contains 'complete') -and $m.complete }
    Write-Json $markerPath ([pscustomobject]@{product='Nebula'; root=$install; complete=$wasComplete})

    # Enable prerequisites without installing the default Ubuntu or asking for a Linux account.
    Log 'Preparing Windows Subsystem for Linux 2...'
    $restart = $false
    foreach ($feature in @('Microsoft-Windows-Subsystem-Linux', 'VirtualMachinePlatform')) {
        $status = Get-WindowsOptionalFeature -Online -FeatureName $feature
        if ($status.State -ne 'Enabled') {
            $result = Enable-WindowsOptionalFeature -Online -FeatureName $feature -All -NoRestart
            if ($result.RestartNeeded) { $restart = $true }
        }
    }
    if ($restart) { Schedule-Resume }
    $ErrorActionPreference = 'Continue'
    $version = & wsl.exe --version 2>&1
    $versionExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($versionExit -ne 0) {
        Native 'wsl.exe' @('--install','--no-distribution','--web-download','--no-launch')
        $version = & wsl.exe --version 2>&1
        if ($LASTEXITCODE -ne 0) { Schedule-Resume }
    }
    Native 'wsl.exe' @('--update','--web-download')

    # .wslconfig is global. Preserve unrelated values and never terminate another distro.
    $wslConfig = Join-Path $env:USERPROFILE '.wslconfig'
    $old = if (Test-Path -LiteralPath $wslConfig) { [IO.File]::ReadAllText($wslConfig) } else { '' }
    $new = Set-IniValues $old 'wsl2' @{memory=('{0}GB' -f $plan.WslMemoryGiB); swap='0'; localhostForwarding='true'}
    if ($old -ne $new) {
        $running = @(Get-WslRunningNames)
        if ($running.Count) { throw ('Close the running WSL distributions ({0}), run "wsl --shutdown", then run Setup again. This is needed to apply the RAM allocation; Setup has not stopped your applications.' -f ($running -join ', ')) }
        if (Test-Path -LiteralPath $wslConfig) { Copy-Item -LiteralPath $wslConfig -Destination ($wslConfig+'.nebula-before-'+(Get-Date -Format yyyyMMddHHmmss)) }
        [IO.File]::WriteAllText($wslConfig, $new, (New-Object Text.UTF8Encoding($false)))
        Native 'wsl.exe' @('--shutdown')
    }

    $distributions = @(Get-WslNames)
    $distroRoot = Join-Path $install 'wsl'
    if ($distributions -contains 'Nebula') {
        $entry = @(Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' | Get-ItemProperty | Where-Object DistributionName -eq 'Nebula')
        if ($entry.Count -ne 1) { throw 'Cannot identify the existing Nebula WSL distribution.' }
        $actual = Normalize-WslPath $entry[0].BasePath
        if ($actual -ine (Normalize-WslPath $distroRoot)) {
            throw 'A different WSL distribution is already named Nebula. Rename it before running Setup; its files have not been changed.'
        }
        $running = @(Get-WslRunningNames)
        if ($running -contains 'Nebula') { throw 'Close Nebula, run "wsl --terminate Nebula", then run Setup again to update it.' }
    } else {
        $dependencies = Read-Json (Join-Path $PSScriptRoot 'dependencies.json')
        $image = Join-Path $install 'ubuntu.wsl'
        Log 'Downloading the verified Ubuntu environment...'
        # Windows curl resumes the file; hashes pin the root filesystem release.
        for ($attempt=0; $attempt -lt 4; $attempt++) {
            & curl.exe --fail --location --retry 3 --connect-timeout 30 --continue-at - --output $image $dependencies.ubuntu.url
            if ($LASTEXITCODE -eq 0) { break }
            if ($attempt -eq 3) { throw 'Ubuntu download interrupted. Run Setup again to resume it.' }
        }
        if ((Get-FileHash -LiteralPath $image -Algorithm SHA256).Hash.ToLowerInvariant() -ne $dependencies.ubuntu.sha256) {
            Remove-Item -LiteralPath $image
            throw 'Ubuntu checksum failed. Run Setup again to download it again.'
        }
        New-Item -ItemType Directory -Path $distroRoot -Force | Out-Null
        Native 'wsl.exe' @('--import','Nebula',$distroRoot,$image,'--version','2')
        Remove-Item -LiteralPath $image
    }

    $planFile = Join-Path $install 'hardware-plan.json'; Write-Json $planFile $plan
    $payloadLinux = ConvertTo-LinuxPath $payload
    $installLinux = ConvertTo-LinuxPath $install
    $planLinux = ConvertTo-LinuxPath $planFile
    Log 'Installing dependencies, compiling Nebula and preparing the official model...'
    Native 'wsl.exe' @('-d','Nebula','-u','root','--','bash',($payloadLinux+'/installer/setup_wsl.sh'),$payloadLinux,$installLinux,$planLinux)
    Native 'wsl.exe' @('-d','Nebula','-u','nebula','--','/home/nebula/app/qwen/build/venv/bin/python','/home/nebula/app/installer/finish_install.py','--plan',$planLinux)

    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'launch.ps1') -Destination (Join-Path $install 'Launch Nebula.ps1') -Force
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'stop.ps1') -Destination (Join-Path $install 'Stop Nebula.ps1') -Force
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Programs')) 'Nebula.lnk'))
    $shortcut.TargetPath = Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'
    $shortcut.Arguments = '-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + (Join-Path $install 'Launch Nebula.ps1') + '"'
    $shortcut.WorkingDirectory = $install
    $shortcut.Description = 'Nebula Chat'
    if ($SetupExecutable) { $shortcut.IconLocation = $SetupExecutable + ',0' }
    $shortcut.Save()
    $stopShortcut = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Programs')) 'Stop Nebula.lnk'))
    $stopShortcut.TargetPath = $shortcut.TargetPath
    $stopShortcut.Arguments = '-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + (Join-Path $install 'Stop Nebula.ps1') + '"'
    $stopShortcut.WorkingDirectory = $install
    $stopShortcut.Description = 'Stop Nebula and release model memory'
    $stopShortcut.Save()
    Copy-Item -LiteralPath (Join-Path ([Environment]::GetFolderPath('Programs')) 'Nebula.lnk') -Destination (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Nebula.lnk') -Force
    Write-Json $markerPath ([pscustomobject]@{product='Nebula';root=$install;complete=$true})
    Remove-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce' -Name 'NebulaSetup' -ErrorAction SilentlyContinue
    Log 'Installation complete. Open Nebula from the desktop or Start menu. The model loads with the first message.'
    exit 0
} catch {
    Write-Output ('SETUP NEEDS ATTENTION: ' + $_.Exception.Message)
    exit 1
}
