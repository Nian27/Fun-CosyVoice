param(
    [string]$Python = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
    [string]$Output = 'E:\AndroidStudioProjects\cosyvoice3-distill-lab\Fun-CosyVoice3-0.5B-2512-RL'
)

$ErrorActionPreference = 'Stop'
$env:HTTP_PROXY = 'http://127.0.0.1:7897'
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'
$Root = $PSScriptRoot

& $Python (Join-Path $Root 'scripts\download_artifacts.py') `
    --manifest (Join-Path $Root 'stage3-manifest.json') `
    --output $Output
if ($LASTEXITCODE -ne 0) {
    throw 'Stage 3 model download failed'
}

$RlModel = Join-Path $Output 'llm.rl.pt'
$LoaderModel = Join-Path $Output 'llm.pt'
if (-not (Test-Path -LiteralPath $LoaderModel)) {
    New-Item -ItemType HardLink -Path $LoaderModel -Target $RlModel | Out-Null
}
Write-Host "Stage 3 model is ready at $Output"
