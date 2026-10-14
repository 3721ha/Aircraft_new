param(
    [string]$PythonExecutable = "python",
    [string]$RunRoot = "results/reproduction_20261010"
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (Test-Path -LiteralPath $RunRoot) { throw "Output already exists. Choose a new -RunRoot: $RunRoot" }
function Invoke-Step {
    param([string[]]$Arguments)
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Reproduction step failed: $($Arguments -join ' ')" }
}
# Preserve the original protocol and its seed/budget defaults; use a NEW output root.
& (Join-Path $PSScriptRoot "run_final_protocol.ps1") -PythonExecutable $PythonExecutable -RunRoot $RunRoot
$proposed = Join-Path $RunRoot "proposed_10seed"
$official = Join-Path $RunRoot "official_10seed"
$seeds = @("11", "22", "33", "44", "55", "66", "77", "88", "99", "111")
$ablationSeeds = @("101", "202", "303", "404", "505", "606", "707", "808", "909", "1001")
Start-Transcript -Path (Join-Path $RunRoot "supplementary_run.log") | Out-Null
try {
    Invoke-Step @("analyze_official_results.py", "--results", $official)
    Invoke-Step @("run_risk_calibration_audit.py", "--proposed-root", $proposed, "--episodes", "40", "--horizon", "30", "--output", (Join-Path $RunRoot "calibration_10seed_fixed"))
    Invoke-Step (@("run_short_recovery_experiments.py", "--seeds") + $seeds + @("--horizon", "30", "--probe-steps", "8", "--output", (Join-Path $RunRoot "short_recovery_paper_protocol_10seed")))
    Invoke-Step (@("run_post_violation_recovery_experiments.py", "--seeds") + $seeds + @("--episodes", "1", "--position-jitter", "0", "--trigger-rule", "S02", "--max-trigger-steps", "12", "--horizon", "30", "--stable-steps", "2", "--output", (Join-Path $RunRoot "post_violation_recovery_10seed_20261008")))
    $exact = Join-Path $RunRoot "timing_oracle_exact_10seed_20261008_v2"
    $varied = Join-Path $RunRoot "timing_oracle_varied_10seed_20261008"
    Invoke-Step (@("run_intervention_timing_oracle_experiments.py", "--seeds") + $seeds + @("--episodes", "1", "--position-jitter", "0", "--oracle-starts", "3", "--oracle-maxiter", "80", "--output", $exact))
    Invoke-Step (@("run_intervention_timing_oracle_experiments.py", "--seeds") + $seeds + @("--episodes", "2", "--position-jitter", "100", "--oracle-starts", "3", "--oracle-maxiter", "80", "--output", $varied))
    Invoke-Step @("analyze_intervention_timing_oracle.py", "--inputs", $exact, $varied, "--output", (Join-Path $RunRoot "timing_oracle_analysis_20261008"))
    Invoke-Step @("run_checkpoint_conflict_traces.py", "--seeds", "11", "--horizon", "30", "--proposed-root", $proposed, "--official-root", $official, "--output", (Join-Path $RunRoot "conflict_traces_seed11"))
    $dynamic = Join-Path $RunRoot "dynamic_priority_activation_10seed"
    Invoke-Step (@("run_dynamic_priority_activation.py", "--seeds") + $ablationSeeds + @("--horizon", "10", "--output", $dynamic))
    Invoke-Step @("analyze_dynamic_priority_activation.py", "--input", (Join-Path $dynamic "per_seed.json"), "--output", (Join-Path $dynamic "statistical_summary"))
    Write-Host "Reproduction completed. Original bundled results are preserved."
    Write-Host "Timing plots are already bundled; this wrapper regenerates pressure-test data and statistical tables."
}
finally { Stop-Transcript | Out-Null }
