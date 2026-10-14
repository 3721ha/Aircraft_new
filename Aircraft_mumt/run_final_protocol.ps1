param(
    [string]$PythonExecutable = "python",
    [string]$RunRoot = "results/final_protocol_20261007"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$runRootPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $RunRoot))
$logPath = Join-Path $runRootPath "run.log"
New-Item -ItemType Directory -Force -Path $runRootPath | Out-Null

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host ""
    Write-Host "===== $Label =====" -ForegroundColor Cyan
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Label (exit code $LASTEXITCODE)"
    }
}

Start-Transcript -Path $logPath -Append | Out-Null
try {
    Invoke-PythonStep "pytest" @(
        "-m", "pytest", "-q"
    )

    $proposedRoot = Join-Path $RunRoot "proposed_10seed"
    Invoke-PythonStep "proposed method (10 seeds)" @(
        "run_trainable_baselines.py",
        "--methods", "proposed_belief_stl_conflict_qp",
        "--seeds", "11", "22", "33", "44", "55", "66", "77", "88", "99", "111",
        "--updates", "100",
        "--episodes", "40",
        "--horizon", "30",
        "--warmstart-epochs", "100",
        "--ppo-epoch", "2",
        "--checkpoint-interval", "5",
        "--output", $proposedRoot
    )

    $officialRoot = Join-Path $RunRoot "official_10seed"
    Invoke-PythonStep "official baselines (10 seeds)" @(
        "run_official_comparison.py",
        "--seeds", "11", "22", "33", "44", "55", "66", "77", "88", "99", "111",
        "--updates", "100",
        "--episodes", "40",
        "--horizon", "30",
        "--ppo-epoch", "5",
        "--checkpoint-interval", "5",
        "--line-search-steps", "10",
        "--safety-bound", "0.1",
        "--proposed-results", $proposedRoot,
        "--output", $officialRoot
    )

    $conflictRoot = Join-Path $RunRoot "conflict_10seed"
    Invoke-PythonStep "conflict ablation (10 seeds)" @(
        "run_conflict_arbitration_experiments.py",
        "--seeds", "101", "202", "303", "404", "505", "606", "707", "808", "909", "1001",
        "--horizon", "30",
        "--initial-feasible-only",
        "--output", $conflictRoot
    )

    Invoke-PythonStep "conflict ablation statistics" @(
        "analyze_conflict_arbitration.py",
        "--input", (Join-Path $conflictRoot "per_seed.json"),
        "--output", (Join-Path $conflictRoot "statistical_summary")
    )

    $componentRoot = Join-Path $RunRoot "component_ablation_10seed"
    Invoke-PythonStep "controlled component ablation (10 seeds)" @(
        "run_component_ablation.py",
        "--seeds", "101", "202", "303", "404", "505", "606", "707", "808", "909", "1001",
        "--horizon", "30",
        "--initial-feasible-only",
        "--output", $componentRoot
    )

    Invoke-PythonStep "controlled component ablation statistics" @(
        "analyze_component_ablation.py",
        "--input", (Join-Path $componentRoot "per_seed.json"),
        "--output", (Join-Path $componentRoot "statistical_summary")
    )

    $jsbsimRoot = Join-Path $RunRoot "jsbsim_10seed"
    Invoke-PythonStep "JSBSim zero-shot validation" @(
        "run_high_fidelity_checkpoint_comparison.py",
        "--backend", "jsbsim",
        "--methods", "proposed", "mappo", "macpo", "mat",
        "--seeds", "11", "22", "33", "44", "55", "66", "77", "88", "99", "111",
        "--episodes", "12",
        "--horizon", "60",
        "--decision-horizon", "30",
        "--proposed-root", $proposedRoot,
        "--official-root", $officialRoot,
        "--output", $jsbsimRoot
    )

    Write-Host ""
    Write-Host "===== All final protocol experiments completed =====" -ForegroundColor Green
    Write-Host "Unified result directory: $runRootPath"
    Write-Host "Main table: $(Join-Path $officialRoot 'summary.json')"
    Write-Host "Conflict ablation: $(Join-Path $conflictRoot 'statistical_summary/summary.json')"
    Write-Host "JSBSim: $(Join-Path $jsbsimRoot 'summary.json')"
}
finally {
    Stop-Transcript | Out-Null
}
