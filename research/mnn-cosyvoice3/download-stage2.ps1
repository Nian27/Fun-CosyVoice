param(
    [string[]]$Only = @()
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = Join-Path $Root '.venv\Scripts\python.exe'
}

$Arguments = @(
    (Join-Path $Root 'scripts\download_artifacts.py'),
    '--manifest', (Join-Path $Root 'stage2-manifest.json'),
    '--output', (Join-Path $Root 'downloads\stage2')
)
foreach ($Name in $Only) {
    $Arguments += @('--only', $Name)
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root 'logs') | Out-Null
$Log = Join-Path $Root ('logs\download-stage2-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
& $Python @Arguments 2>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) {
    throw "Stage 2 artifact download failed. See $Log"
}
