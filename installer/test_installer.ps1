Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'hardware.ps1')
$payload = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$count = 0
function Assert($Condition, [string]$Message) { if (!$Condition) { throw $Message }; $script:count++ }
$hardware = [pscustomobject]@{RamBytes=128GB;AvailableRamBytes=120GB;PhysicalCores=6;LogicalCores=12;
    Gpu=[pscustomobject]@{Uuid='GPU-test';TotalBytes=12GB;FreeBytes=12GB};
    Drives=@([pscustomobject]@{Root='C:\';FreeBytes=300GB;Ssd=$true},[pscustomobject]@{Root='E:\';FreeBytes=600GB;Ssd=$false})}
foreach ($g in @(8,12,16,24)) { foreach ($r in @(32,64,96,128)) {
    $hardware.Gpu.TotalBytes=$g*1GB;$hardware.Gpu.FreeBytes=$g*1GB
    $hardware.RamBytes=$r*1GB;$hardware.AvailableRamBytes=($r-2)*1GB
    $plan=Get-InstallPlan $hardware $payload
    Assert ($plan.Profile.vram_gib -eq $g -and $plan.Profile.ram_gib -eq $r) 'Wrong tier'
    Assert ($plan.Profile.ram_experts+$plan.Profile.ssd_experts -eq 512) 'Wrong expert split'
    Assert ($plan.CanInstall) ('Unexpected profile error: '+($plan.RequiredActions -join ' '))
    Assert ($plan.InstallRoot -eq 'C:\Nebula') 'Selected HDD instead of SSD'
    Assert ($plan.CpuThreads -eq 6) 'Wrong physical CPU thread selection'
} }
$hardware.Gpu.TotalBytes=20GB;$hardware.RamBytes=80GB
$p=Get-ApprovedProfile $hardware $payload
Assert ($p.vram_gib -eq 16 -and $p.ram_gib -eq 64) 'Independent floor selection failed'
$hardware.Gpu.TotalBytes=12282MB
Assert ((Get-ApprovedProfile $hardware $payload).vram_gib -eq 12) 'Nominal NVIDIA size failed'
$hardware.Drives[0].FreeBytes=1GB;$hardware.AvailableRamBytes=1GB;$hardware.Gpu.FreeBytes=1MB
$plan=Get-InstallPlan $hardware $payload
Assert (!$plan.CanInstall -and $plan.RequiredActions.Count -eq 3) 'Expected exact disk/RAM/VRAM deficits'
$old="[wsl2]`r`nmemory=40GB`r`n# keep me`r`nprocessors=12`r`n[experimental]`r`nsparseVhd=true`r`n"
$new=Set-IniValues $old 'wsl2' @{memory='112GB';swap='0';localhostForwarding='true'}
Assert ($new.Contains('processors=12') -and $new.Contains('# keep me') -and $new.Contains('sparseVhd=true')) 'INI lost unrelated settings'
Assert ($new.Contains('memory=112GB') -and !$new.Contains('memory=40GB')) 'INI failed replacement'
Assert ((Set-IniValues $new 'wsl2' @{memory='112GB';swap='0';localhostForwarding='true'}) -eq $new) 'INI update is not idempotent'
Assert ((ConvertTo-LinuxPath 'C:\A directory\Nebula') -eq '/mnt/c/A directory/Nebula') 'Path conversion failed'
Assert ((Normalize-WslPath '\\?\C:\Nebula\wsl') -eq 'C:\Nebula\wsl') 'Extended WSL path normalization failed'
$temporary=Join-Path ([IO.Path]::GetTempPath()) ('nebula-resume-test-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path (Join-Path $temporary 'cache') -Force | Out-Null
$testFile=Join-Path $temporary 'cache/part.download'
try {
    [IO.File]::WriteAllBytes($testFile,(New-Object byte[] 65536))
    Assert ((Get-ManagedInstallBytes $temporary) -ge 65536) 'Interrupted download was not credited for resume'
} finally {
    Remove-Item -LiteralPath $testFile
    Remove-Item -LiteralPath (Join-Path $temporary 'cache')
    Remove-Item -LiteralPath $temporary
}
Write-Output "$count installer hardware/configuration assertions passed. No system settings changed."
