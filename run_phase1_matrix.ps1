param(
    [int]$Budget = 2000000,
    [int]$SessionSeconds = 1000,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$conditions = @(
    "iid_clm",
    "ordered_clm",
    "trial_only_rl",
    "yoked_caregiver_rl",
    "contingent_caregiver_rl"
)
$seeds = @(5000, 5001, 5002)
$cache = "artifacts/phase1_caregiver_cache.json"
$tokenizer = "artifacts/phase1_vocab.model"

& nvidia-smi

Write-Host "Running PPO optimizer gate"
& $Python phase1_vocab.py `
    --overfit-test `
    --condition contingent_caregiver_rl `
    --seed 4900 `
    --caregiver-cache $cache `
    --tokenizer $tokenizer
if ($LASTEXITCODE -ne 0) { throw "Phase-1 PPO gate failed" }

foreach ($condition in $conditions) {
    foreach ($seed in $seeds) {
        $runDir = "results/phase1/phase1-$condition-$seed"
        $checkpoint = Join-Path $runDir "latest.pt"
        do {
            $args = @(
                "phase1_vocab.py",
                "--condition", $condition,
                "--seed", $seed,
                "--budget", $Budget,
                "--session-seconds", $SessionSeconds,
                "--caregiver-cache", $cache,
                "--tokenizer", $tokenizer
            )
            if (Test-Path $checkpoint) {
                $args += @("--resume", $checkpoint)
            }
            & $Python @args
            if ($LASTEXITCODE -ne 0) { throw "Run failed: $condition seed $seed" }
            $resultPath = Join-Path $runDir "result.json"
            $complete = $false
            if (Test-Path $resultPath) {
                $result = Get-Content $resultPath -Raw | ConvertFrom-Json
                $complete = [bool]$result.complete
            }
        } until ($complete)
    }
}

Write-Host "Phase-1 development matrix complete"
