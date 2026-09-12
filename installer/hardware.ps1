Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Read-Json([string]$Path) {
    return [IO.File]::ReadAllText($Path) | ConvertFrom-Json
}

function Write-Json([string]$Path, $Value) {
    $temporary = $Path + '.tmp'
    [IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 20), (New-Object Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Get-LowerTier([double]$Bytes, [int[]]$Tiers, [long]$Tolerance = 0) {
    $eligible = @($Tiers | Where-Object { $Bytes + $Tolerance -ge $_ * 1GB })
    if (!$eligible.Count) { throw ('At least {0} GiB required; missing {1:N2} GiB.' -f $Tiers[0], (($Tiers[0] * 1GB - $Bytes) / 1GB)) }
    return ($eligible | Measure-Object -Maximum).Maximum
}

function Get-WslNames {
    $key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss'
    if (!(Test-Path $key)) { return @() }
    return @(Get-ChildItem $key | Get-ItemProperty | Where-Object { $_.PSObject.Properties.Name -contains 'DistributionName' } | ForEach-Object DistributionName)
}

function Get-WslRunningNames {
    $names = @(Get-WslNames)
    if (!$names.Count) { return @() }
    $raw = @((& wsl.exe --list --running --quiet) | ForEach-Object { ($_ -replace "`0",'').Trim() })
    # Empty installations may print localized explanatory text. Match only
    # registered distribution names, never interpret that text as a process.
    return @($names | Where-Object { $raw -contains $_ })
}

function Normalize-WslPath([string]$Path) {
    if ($Path.StartsWith('\\?\')) { $Path = $Path.Substring(4) }
    return [IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function Get-ManagedInstallBytes([string]$Root) {
    if (!('NebulaDiskSize' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class NebulaDiskSize {
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  static extern uint GetCompressedFileSizeW(string path, out uint high);
  public static long Allocated(string path) {
    uint high; uint low=GetCompressedFileSizeW(path,out high);
    if(low==uint.MaxValue && Marshal.GetLastWin32Error()!=0)return 0;
    return (long)high*4294967296L+low;
  }
}
'@
    }
    $sum = 0L
    foreach ($dir in @('model','cache')) {
        $path = Join-Path $Root $dir
        if (Test-Path -LiteralPath $path) {
            Get-ChildItem -LiteralPath $path -File -Recurse | ForEach-Object {
                $sum += [NebulaDiskSize]::Allocated($_.FullName)
            }
        }
    }
    $vhdx = Join-Path $Root 'wsl/ext4.vhdx'
    if (Test-Path -LiteralPath $vhdx) {
        $sum += [Math]::Min(20GB, [NebulaDiskSize]::Allocated($vhdx))
    }
    return $sum
}

function Get-ApprovedProfile($Hardware, [string]$Payload) {
    $gpuTier = Get-LowerTier $Hardware.Gpu.TotalBytes @(8, 12, 16, 24) (64MB)
    $ramTier = Get-LowerTier $Hardware.RamBytes @(32, 64, 96, 128)
    $table = Read-Json (Join-Path $Payload 'qwen/hardware_profiles.json')
    return @($table.profiles | Where-Object { $_.vram_gib -eq $gpuTier -and $_.ram_gib -eq $ramTier })[0]
}

function Get-WindowsHardware {
    $os = Get-CimInstance Win32_OperatingSystem
    $system = Get-CimInstance Win32_ComputerSystem
    $processors = @(Get-CimInstance Win32_Processor)
    if (![Environment]::Is64BitOperatingSystem -or $processors[0].Architecture -ne 9) { throw 'Nebula requires x86-64 Windows and an NVIDIA GPU.' }
    if ([int]$os.BuildNumber -lt 19045) { throw 'Update Windows to Windows 10 build 19045 or later, then run Setup again.' }
    if (!$system.HypervisorPresent -and !($processors | Where-Object VirtualizationFirmwareEnabled)) {
        throw 'Enable hardware virtualization in the BIOS/UEFI, restart Windows, then run Setup again.'
    }
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (!$smi) {
        $path = Join-Path $env:ProgramFiles 'NVIDIA Corporation/NVSMI/nvidia-smi.exe'
        if (!(Test-Path -LiteralPath $path)) { throw 'Install the current NVIDIA Windows driver, restart Windows, then run Setup again.' }
    } else { $path = $smi.Source }
    $raw = & $path '--query-gpu=index,uuid,name,memory.total,memory.free,compute_cap,driver_version' '--format=csv,noheader,nounits'
    if ($LASTEXITCODE -ne 0) { throw 'NVIDIA hardware detection failed. Update the NVIDIA Windows driver, then run Setup again.' }
    $gpus = @($raw | ConvertFrom-Csv -Header Index,Uuid,Name,Total,Free,Compute,Driver | ForEach-Object {
        [pscustomobject]@{ Index = [int]$_.Index; Uuid = $_.Uuid.Trim(); Name = $_.Name.Trim();
            TotalBytes = [long]([double]$_.Total * 1MB); FreeBytes = [long]([double]$_.Free * 1MB);
            Compute = $_.Compute.Trim(); Driver = $_.Driver.Trim() }
    })
    if (!$gpus.Count) { throw 'No supported NVIDIA GPU was detected.' }
    $gpu = @($gpus | Sort-Object TotalBytes, FreeBytes -Descending)[0]
    if ([version]$gpu.Driver -lt [version]'570.65') { throw 'Update the NVIDIA Windows driver to 570.65 or later, restart Windows, then run Setup again.' }
    $ram = (Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum
    if (!$ram) { throw 'Windows could not report installed physical RAM.' }
    $physical = @(Get-PhysicalDisk -ErrorAction SilentlyContinue)
    $drives = @(Get-Partition -ErrorAction SilentlyContinue | Where-Object DriveLetter | ForEach-Object {
        $partition = $_; $disk = $partition | Get-Disk; $volume = $partition | Get-Volume
        $pd = @($physical | Where-Object { [string]$_.DeviceId -eq [string]$disk.Number })
        $ssd = ([string]$disk.BusType -eq 'NVMe') -or @($pd | Where-Object { [string]$_.MediaType -eq 'SSD' }).Count -gt 0
        if ($volume.FileSystem -eq 'NTFS' -and $disk.BusType -ne 'USB') {
            [pscustomobject]@{ Root = ($partition.DriveLetter + ':\'); FreeBytes = [long]$volume.SizeRemaining;
                TotalBytes = [long]$volume.Size; Ssd = [bool]$ssd; Model = $disk.FriendlyName }
        }
    })
    return [pscustomobject]@{ RamBytes = [long]$ram; AvailableRamBytes = [long]$os.FreePhysicalMemory * 1024;
        PhysicalCores = [int](($processors | Measure-Object NumberOfCores -Sum).Sum);
        LogicalCores = [int](($processors | Measure-Object NumberOfLogicalProcessors -Sum).Sum);
        Gpu = $gpu; Drives = $drives; WindowsBuild = [int]$os.BuildNumber }
}

function Get-InstallPlan($Hardware, [string]$Payload, [string]$ExistingRoot = '') {
    $profile = Get-ApprovedProfile $Hardware $Payload
    $budgets = Read-Json (Join-Path $Payload 'installer/budgets.json')
    $budget = @($budgets.profiles | Where-Object { $_.vram_gib -eq $profile.vram_gib -and $_.ram_gib -eq $profile.ram_gib })[0]
    $ssds = @($Hardware.Drives | Where-Object Ssd | Sort-Object FreeBytes -Descending)
    if (!$ssds.Count) { throw 'No internal NTFS SSD was detected for the model and WSL installation.' }
    $drive = $null
    if ($ExistingRoot) {
        $rootDrive = [IO.Path]::GetPathRoot($ExistingRoot)
        $drive = @($ssds | Where-Object Root -eq $rootDrive | Select-Object -First 1)
        if (!$drive.Count) { throw 'The SSD used by this Nebula installation is not available.' }
        $drive = $drive[0]
    } else {
        $preferred = @($ssds | Where-Object { $_.Root -eq ($env:SystemDrive + '\') -and $_.FreeBytes -ge $budgets.peak_install_bytes })
        $drive = if ($preferred.Count) { $preferred[0] } else { $ssds[0] }
        $ExistingRoot = Join-Path $drive.Root 'Nebula'
    }
    $errors = New-Object 'Collections.Generic.List[string]'
    $diskNeeded = [long]$budgets.peak_install_bytes
    # Credit only files inside the exact managed destination. No scans of unrelated model folders.
    $marker = Join-Path $ExistingRoot '.nebula-install.json'
    if (Test-Path -LiteralPath $marker) {
        $owned = Read-Json $marker
        if ($owned.product -ne 'Nebula' -or $owned.root -ne $ExistingRoot) { throw 'Installation marker does not match its directory.' }
        $modelDir = Join-Path $ExistingRoot 'model'
        $completed = 0L
        if (Test-Path -LiteralPath $modelDir) {
            Get-ChildItem -LiteralPath $modelDir -File -Filter '*.bin' | ForEach-Object { $completed += $_.Length }
        }
        $diskNeeded = [Math]::Max(2GB, $diskNeeded - (Get-ManagedInstallBytes $ExistingRoot))
        if ($completed -eq $budgets.model_final_bytes) { $diskNeeded = 2GB }
        if ($owned.PSObject.Properties.Name -contains 'complete' -and $owned.complete) { $diskNeeded = 2GB }
    } elseif ((Test-Path -LiteralPath $ExistingRoot) -and @(Get-ChildItem -LiteralPath $ExistingRoot -Force).Count) {
        $errors.Add("$ExistingRoot already contains files from another application. Move that folder, then run Setup again.")
    }
    $missing = $diskNeeded - [long]$drive.FreeBytes
    if ($missing -gt 0) { $errors.Add(('Free another {0:N2} GiB on {1}, then run Setup again.' -f ($missing/1GB), $drive.Root)) }
    if ($drive.Root -ne ($env:SystemDrive + '\')) {
        $system = @($Hardware.Drives | Where-Object Root -eq ($env:SystemDrive + '\'))
        if ($system.Count -and $system[0].FreeBytes -lt 2GB) { $errors.Add(('Free another {0:N2} GiB on the Windows system drive for WSL and Setup, then retry.' -f ((2GB-$system[0].FreeBytes)/1GB))) }
    }
    $missing = [long]$budget.ram_required_free - [long]$Hardware.AvailableRamBytes
    if ($missing -gt 0) { $errors.Add(('Close applications to release another {0:N2} GiB of RAM, then run Setup again.' -f ($missing/1GB))) }
    $missing = [long]$budget.gpu_required_free - [long]$Hardware.Gpu.FreeBytes
    if ($missing -gt 0) { $errors.Add(('Close GPU applications to release another {0:N2} GiB of VRAM, then run Setup again.' -f ($missing/1GB))) }
    return [pscustomobject]@{ Version = 1; Hardware = $Hardware; Profile = $profile; Budget = $budget;
        InstallRoot = $ExistingRoot; DiskRequiredFree = $diskNeeded;
        CpuThreads = [Math]::Max(1, [Math]::Min(28, $Hardware.PhysicalCores));
        WslMemoryGiB = [int][Math]::Floor($Hardware.RamBytes / 1GB) - $profile.windows_reserve_gib;
        CanInstall = ($errors.Count -eq 0); RequiredActions = @($errors.ToArray()) }
}

function Set-IniValues([string]$Text, [string]$Section, [hashtable]$Values) {
    $result = New-Object 'Collections.Generic.List[string]'
    $inside = $false; $found = $false; $seen = @{}
    foreach ($line in ($Text -split "`r?`n")) {
        if ($line -match '^\s*\[([^\]]+)\]\s*$') {
            if ($inside) { foreach ($key in $Values.Keys) { if (!$seen.ContainsKey($key)) { $result.Add("$key=$($Values[$key])") } } }
            $inside = $Matches[1] -ieq $Section
            if ($inside) { $found = $true }
        }
        if ($inside -and $line -match '^\s*([^#;=]+?)\s*=') {
            $key = $Matches[1].Trim()
            if ($Values.ContainsKey($key)) {
                if (!$seen.ContainsKey($key)) { $result.Add("$key=$($Values[$key])"); $seen[$key] = $true }
                continue
            }
        }
        $result.Add($line)
    }
    if (!$found) { $result.Add("[$Section]") }
    if ($inside -or !$found) { foreach ($key in $Values.Keys) { if (!$seen.ContainsKey($key)) { $result.Add("$key=$($Values[$key])") } } }
    return ($result -join "`r`n").Trim() + "`r`n"
}

function ConvertTo-LinuxPath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if ($full -notmatch '^([A-Za-z]):\\') { throw 'Installation requires a local drive letter.' }
    return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $full.Substring(3).Replace('\','/')
}
