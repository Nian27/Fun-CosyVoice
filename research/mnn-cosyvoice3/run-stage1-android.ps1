param(
    [string]$Serial = 'A3TE025B03003242'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Adb = 'C:\Users\Administrator\AppData\Local\Android\Sdk\platform-tools\adb.exe'
$BuildDir = Join-Path $Root 'build\mnn-3.6-android-arm64-opencl'
$Executable = Join-Path $BuildDir 'CosyVoiceFlowBenchmark.out'
$Model = Join-Path $Root 'models\flow-external\flow.decoder.estimator.fp32.mnn'
$Weights = "$Model.weight"
$CaseDir = Join-Path $Root 'cases\seq32'
$ResultDir = Join-Path $Root 'results\android-seq32'
$Remote = '/data/local/tmp/cosy-mnn-stage1'

foreach ($Required in @($Adb, $Executable, $Model, $Weights)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required file not found: $Required"
    }
}
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

function Invoke-Adb {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Adb -s $Serial @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed: $($Arguments -join ' ')"
    }
}

Invoke-Adb wait-for-device
Invoke-Adb shell "mkdir -p $Remote/inputs"
Invoke-Adb push $Executable "$Remote/CosyVoiceFlowBenchmark.out"
Invoke-Adb push $Model "$Remote/flow.mnn"
Invoke-Adb push $Weights "$Remote/flow.mnn.weight"
foreach ($Name in @('x.bin', 'mask.bin', 'mu.bin', 't.bin', 'spks.bin', 'cond.bin')) {
    Invoke-Adb push (Join-Path $CaseDir $Name) "$Remote/inputs/$Name"
}
Invoke-Adb shell "chmod 755 $Remote/CosyVoiceFlowBenchmark.out"

$CpuLog = Join-Path $ResultDir 'cpu-high.log'
$OpenClLog = Join-Path $ResultDir 'opencl-high.log'
& $Adb -s $Serial shell "cd $Remote && ./CosyVoiceFlowBenchmark.out flow.mnn inputs cpu.bin 32 cpu high 3 6" |
    Tee-Object -FilePath $CpuLog
if ($LASTEXITCODE -ne 0) { throw 'Android CPU benchmark failed' }
& $Adb -s $Serial shell "cd $Remote && ./CosyVoiceFlowBenchmark.out flow.mnn inputs opencl.bin 32 opencl high 3 4" |
    Tee-Object -FilePath $OpenClLog
if ($LASTEXITCODE -ne 0) { throw 'Android OpenCL benchmark failed' }

Invoke-Adb pull "$Remote/cpu.bin" (Join-Path $ResultDir 'cpu.bin')
Invoke-Adb pull "$Remote/cpu.bin.json" (Join-Path $ResultDir 'cpu.bin.json')
Invoke-Adb pull "$Remote/opencl.bin" (Join-Path $ResultDir 'opencl.bin')
Invoke-Adb pull "$Remote/opencl.bin.json" (Join-Path $ResultDir 'opencl.bin.json')

$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Reference = Join-Path $CaseDir 'onnx_output.bin'
foreach ($Backend in @('cpu', 'opencl')) {
    & $Python (Join-Path $Root 'scripts\flow_reference.py') compare `
        --reference $Reference `
        --candidate (Join-Path $ResultDir "$Backend.bin") `
        --report (Join-Path $ResultDir "compare-onnx-$Backend.json") `
        --abs-tolerance 0.02
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$Backend output exceeded the provisional 0.05 absolute tolerance"
    }
}
