param(
    [string]$Model = "models\best_pytorch_model_final.pth",
    [string]$ModelName = "efficientnet_b4",
    [Alias("Test")]
    [string]$TestSplit = "splits\test.txt",
    [Alias("Train")]
    [string]$TrainSplit = "splits\train.txt",
    [Alias("Val")]
    [string]$ValSplit = "splits\val.txt",
    [string]$OutDir = "",
    [ValidateSet("checkpoint", "eer", "fixed")]
    [string]$ThresholdMode = "checkpoint",
    [double]$FixedThreshold = 0.5,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 0,
    [string]$PythonExe = "python",
    [string[]]$ExternalSplit = @(),
    [string[]]$ExternalName = @(),
    [string[]]$ExternalDir = @(),
    [switch]$Robustness,
    [switch]$SkipDfdc,
    [switch]$FrequencyBranch,
    [ValidateSet("auto", "none", "laplacian", "fft")]
    [string]$FrequencyMode = "auto",
    [int]$FrequencyFeatures = 128,
    [switch]$Cpu,
    [int]$MaxSamples = 0
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($PythonExe -like "*deepfake_env*Scripts*python.exe" -and $env:CONDA_PREFIX) {
    $CondaPython = Join-Path $env:CONDA_PREFIX "python.exe"
    if (Test-Path $CondaPython) {
        Write-Warning "Detected broken local deepfake_env Python. Using active conda Python instead: $CondaPython"
        $PythonExe = $CondaPython
    }
}

Write-Host "Using Python: $PythonExe"
& $PythonExe -c "import sys, encodings; print(sys.executable)" | Write-Host

$argsList = @(
    "evaluate_comprehensive.py",
    "--model", $Model,
    "--model-name", $ModelName,
    "--test", $TestSplit,
    "--train", $TrainSplit,
    "--val", $ValSplit,
    "--threshold-mode", $ThresholdMode,
    "--fixed-threshold", "$FixedThreshold",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers"
)

if ($OutDir -ne "") {
    $argsList += @("--out", $OutDir)
}

if ($Robustness) {
    $argsList += "--robustness"
}

if ($SkipDfdc) {
    $argsList += "--skip-dfdc"
}

if ($FrequencyBranch) {
    $argsList += "--frequency-branch"
}

$argsList += @("--frequency-mode", $FrequencyMode)
$argsList += @("--frequency-features", "$FrequencyFeatures")

if ($ExternalSplit.Count -gt 0) {
    if ($ExternalName.Count -gt 0 -and $ExternalName.Count -ne $ExternalSplit.Count) {
        throw "ExternalName count must match ExternalSplit count, or omit ExternalName."
    }

    for ($i = 0; $i -lt $ExternalSplit.Count; $i++) {
        $splitValue = $ExternalSplit[$i]
        if ($ExternalName.Count -gt 0) {
            $splitValue = "$($ExternalName[$i]):$splitValue"
        }
        $argsList += @("--external-split", $splitValue)
    }
}

foreach ($dir in $ExternalDir) {
    $argsList += @("--external-dir", $dir)
}

if ($Cpu) {
    $argsList += "--cpu"
}

if ($MaxSamples -gt 0) {
    $argsList += @("--max-samples", "$MaxSamples")
}

& $PythonExe @argsList
