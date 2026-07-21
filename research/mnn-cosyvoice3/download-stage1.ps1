param(
    [string[]]$Only = @()
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    $PythonCommand = Get-Command python -ErrorAction Stop
    $Python = $PythonCommand.Source
}

$Arguments = @(
    (Join-Path $Root 'scripts\download_artifacts.py'),
    '--manifest', (Join-Path $Root 'stage1-manifest.json'),
    '--output', (Join-Path $Root 'downloads')
)
foreach ($Name in $Only) {
    $Arguments += @('--only', $Name)
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root 'logs') | Out-Null
$Log = Join-Path $Root ('logs\download-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
& $Python @Arguments 2>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) {
    throw "Artifact download failed. See $Log"
}
