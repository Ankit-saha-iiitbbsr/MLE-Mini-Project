<#
.SYNOPSIS
    Run the DefectVision pipeline end to end (M2 -> M5).

.DESCRIPTION
    The Windows equivalent of `make all`. Runs each stage in order and stops at
    the first failure, so a broken data gate does not silently produce a model
    trained on bad data.

    The retraining check is exempt from that rule: it exits 10 when the trigger
    legitimately fires, which is a result, not an error.

.PARAMETER Source
    Data source: kaggle (default, per params.yaml), synthetic, or local.

.PARAMETER SkipTraining
    Reuse existing models and jump to the monitoring stages.

.EXAMPLE
    .\scripts\run_pipeline.ps1

.EXAMPLE
    .\scripts\run_pipeline.ps1 -Source synthetic
#>
[CmdletBinding()]
param(
    [ValidateSet('kaggle', 'synthetic', 'local')]
    [string]$Source,

    [switch]$SkipTraining
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Prefer the project venv; fall back to whatever python is on PATH.
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Warning "No venv at .venv - falling back to 'python' on PATH. Run 'make setup' to create one."
    $python = 'python'
}

function Invoke-Stage {
    param(
        [Parameter(Mandatory)][string]$Title,
        [Parameter(Mandatory)][string[]]$Arguments,
        [int[]]$AllowExitCodes = @(0)
    )

    Write-Host ''
    Write-Host ('=' * 74) -ForegroundColor DarkCyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ('=' * 74) -ForegroundColor DarkCyan

    $started = Get-Date
    & $python -m defectvision.cli @Arguments
    $code = $LASTEXITCODE
    $elapsed = ((Get-Date) - $started).TotalSeconds

    if ($AllowExitCodes -notcontains $code) {
        Write-Host "FAILED after $([math]::Round($elapsed, 1))s (exit $code)" -ForegroundColor Red
        exit $code
    }
    Write-Host "OK in $([math]::Round($elapsed, 1))s" -ForegroundColor Green
    return $code
}

$overall = Get-Date

# --- M2 ---------------------------------------------------------------------
$dataArgs = @('data')
if ($Source) { $dataArgs += @('--source', $Source) }
Invoke-Stage -Title 'M2 | Data: acquire, validate, split' -Arguments $dataArgs | Out-Null

# --- M3 ---------------------------------------------------------------------
if (-not $SkipTraining) {
    Invoke-Stage -Title 'M3 | Train every configured arm' -Arguments @('train', '--all', '--no-compare') | Out-Null
    Invoke-Stage -Title 'M3 | Compare, gate, promote' -Arguments @('compare') | Out-Null
} else {
    Write-Host "`nSkipping training (-SkipTraining)" -ForegroundColor Yellow
}

# --- M5 ---------------------------------------------------------------------
Invoke-Stage -Title 'M5 | Build the drift reference baseline' -Arguments @('reference-stats') | Out-Null
Invoke-Stage -Title 'M5 | Simulate distribution shift' -Arguments @('simulate-drift') | Out-Null
Invoke-Stage -Title 'M5 | Monitoring report' -Arguments @('monitor') | Out-Null

# Exit 10 means the retraining trigger fired -- a valid outcome, not a failure.
$retrainCode = Invoke-Stage -Title 'M5 | Retraining trigger' -Arguments @('check-retrain') -AllowExitCodes @(0, 10)

# --- Summary ----------------------------------------------------------------
$total = ((Get-Date) - $overall).TotalMinutes
Write-Host ''
Write-Host ('=' * 74) -ForegroundColor DarkCyan
Write-Host "  Pipeline complete in $([math]::Round($total, 1)) minutes" -ForegroundColor Cyan
Write-Host ('=' * 74) -ForegroundColor DarkCyan

if ($retrainCode -eq 10) {
    Write-Host 'Retraining trigger FIRED - see reports/retraining_decision.json' -ForegroundColor Yellow
} else {
    Write-Host 'Retraining trigger did not fire.' -ForegroundColor Green
}

Write-Host "`nArtifacts:"
@(
    'reports/validation_report.json     M2 data quality gates',
    'data/processed/dataset_card.json   M2 dataset version',
    'reports/model_comparison.md        M3 model comparison',
    'models/production/model_bundle.pt  M4 deployable model',
    'reports/drift_report.md            M5 drift analysis',
    'reports/retraining_decision.json   M5 retraining decision'
) | ForEach-Object { Write-Host "  $_" }

Write-Host "`nNext: start the API with" -NoNewline
Write-Host "  $python -m defectvision.cli serve" -ForegroundColor White
