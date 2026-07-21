param(
    [string]$Converter = ''
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Converter)) {
    $Converter = Join-Path $Root 'build\mnn-3.6-converter\Release\MNNConvert.exe'
}
$ModelDir = Join-Path $Root 'models\hift-split'
if (-not (Test-Path -LiteralPath $Converter)) {
    throw "MNN converter not found: $Converter"
}
foreach ($Name in @('hift-f0', 'hift-core')) {
    $Onnx = Join-Path $ModelDir "$Name.fp32.onnx"
    $Mnn = Join-Path $ModelDir "$Name.fp32.mnn"
    if (-not (Test-Path -LiteralPath $Onnx)) {
        throw "ONNX model not found: $Onnx"
    }
    & $Converter -f ONNX --modelFile $Onnx --MNNModel $Mnn --bizCode CosyVoice3
    if ($LASTEXITCODE -ne 0) {
        throw "MNN conversion failed: $Name"
    }
}
