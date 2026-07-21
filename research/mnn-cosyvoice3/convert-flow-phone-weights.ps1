param(
    [ValidateSet('fp16', 'int8')]
    [string]$Mode = 'fp16'
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Converter = Join-Path $Root 'build\mnn-3.6-converter\Release\MNNConvert.exe'
$Source = Join-Path $Root 'models\flow-cfg-student-2step\flow.decoder.estimator.cfg-student-2step.batch1.fp32.onnx'
$OutputDir = Join-Path $Root "models\flow-cfg-student-2step-$Mode"
$Output = Join-Path $OutputDir "flow.cfg-student-2step.batch1.$Mode.mnn"

foreach ($Required in @($Converter, $Source)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required file not found: $Required"
    }
}
New-Item -ItemType Directory -Force $OutputDir | Out-Null

$Arguments = @(
    '-f', 'ONNX',
    '--modelFile', $Source,
    '--MNNModel', $Output,
    '--bizCode', 'MNN',
    '--saveExternalData',
    '--optimizeLevel', '1'
)
if ($Mode -eq 'fp16') {
    $Arguments += '--fp16'
} else {
    $Arguments += @(
        '--weightQuantBits', '8',
        '--weightQuantBlock', '128',
        '--weightQuantScaleBit', '16'
    )
}

& $Converter @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "MNN Flow $Mode conversion failed"
}

$Files = Get-ChildItem -LiteralPath $OutputDir -File |
    Select-Object Name, Length, @{Name = 'Sha256'; Expression = { (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }}
$Files | Format-Table -AutoSize
$Files | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDir 'manifest.json') -Encoding utf8
