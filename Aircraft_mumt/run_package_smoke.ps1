param(
    [string]$PythonExecutable = "python",
    [string]$RunRoot = "results/package_smoke"
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (Test-Path -LiteralPath $RunRoot) { throw "Output already exists. Choose a new -RunRoot: $RunRoot" }
New-Item -ItemType Directory -Path $RunRoot | Out-Null
function Invoke-Step {
    param([string[]]$Arguments)
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Smoke step failed: $($Arguments -join ' ')" }
}
Start-Transcript -Path (Join-Path $RunRoot "smoke.log") | Out-Null
try {
    Invoke-Step @("-m", "pytest", "-q")
    $proposed = Join-Path $RunRoot "proposed"
    $official = Join-Path $RunRoot "official"
    Invoke-Step @("run_trainable_baselines.py", "--methods", "proposed_belief_stl_conflict_qp", "--seeds", "11", "--updates", "2", "--episodes", "2", "--horizon", "6", "--warmstart-epochs", "5", "--checkpoint-interval", "1", "--output", $proposed)
    Invoke-Step @("run_official_comparison.py", "--seeds", "11", "--updates", "2", "--episodes", "2", "--horizon", "6", "--ppo-epoch", "2", "--checkpoint-interval", "1", "--line-search-steps", "3", "--safety-bound", "0.1", "--proposed-results", $proposed, "--output", $official)
    Invoke-Step @("run_high_fidelity_checkpoint_comparison.py", "--backend", "jsbsim", "--methods", "proposed", "mappo", "macpo", "mat", "--seeds", "11", "--episodes", "1", "--horizon", "3", "--decision-horizon", "6", "--proposed-root", $proposed, "--official-root", $official, "--output", (Join-Path $RunRoot "jsbsim"))
    Invoke-Step @("run_checkpoint_conflict_traces.py", "--seeds", "11", "--horizon", "3", "--proposed-root", $proposed, "--official-root", $official, "--output", (Join-Path $RunRoot "traces"))
    Write-Host "Smoke completed: seven policies, JSBSim checkpoint validation and per-step traces."
    Write-Host "These small runs verify execution only; they are not paper results."
}
finally { Stop-Transcript | Out-Null }
