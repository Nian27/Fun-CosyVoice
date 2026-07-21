param(
    [string]$CosyVoiceSource = 'E:\AndroidStudioProjects\CosyVoice-main',
    [string]$ReferenceWav = 'E:\AndroidStudioProjects\legado-archive-v3-3.26.06071119\app\src\cosytest\assets\cosyvoice\zero_shot_prompt.wav'
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = Join-Path $Root '.venv-export\Scripts\python.exe'
$Weights = Join-Path $Root 'downloads\stage2\hift.pt'
$Output = Join-Path $Root 'models\hift-split'
foreach ($Required in @($Python, $Weights, $ReferenceWav)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required file not found: $Required"
    }
}
New-Item -ItemType Directory -Force -Path $Output | Out-Null
& $Python (Join-Path $Root 'scripts\export_hift_split.py') `
    --cosyvoice-source $CosyVoiceSource `
    --weights $Weights `
    --reference-wav $ReferenceWav `
    --output $Output
if ($LASTEXITCODE -ne 0) {
    throw 'HiFT split export or ONNX validation failed'
}
