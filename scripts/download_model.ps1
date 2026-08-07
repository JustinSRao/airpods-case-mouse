<#
.SYNOPSIS
    Downloads the MediaPipe hand landmark model bundle into models\.

.DESCRIPTION
    The bundle is ~7.5 MB and is intentionally not committed to git. Run this
    once after cloning. The float16 variant is used: it is half the size of
    the float32 build with no measurable accuracy loss for this use case.

.EXAMPLE
    .\scripts\download_model.ps1
#>
[CmdletBinding()]
param(
    [string]$Variant = "float16",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modelsDir = Join-Path $repoRoot "models"
$target = Join-Path $modelsDir "hand_landmarker.task"
$url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/$Variant/1/hand_landmarker.task"

if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}

if ((Test-Path $target) -and (-not $Force)) {
    $size = [math]::Round((Get-Item $target).Length / 1MB, 2)
    Write-Host "Model already present ($size MB): $target"
    Write-Host "Use -Force to re-download."
    exit 0
}

Write-Host "Downloading $Variant hand_landmarker.task ..."
Invoke-WebRequest -Uri $url -OutFile $target

$size = [math]::Round((Get-Item $target).Length / 1MB, 2)
Write-Host "Saved $size MB to $target"
