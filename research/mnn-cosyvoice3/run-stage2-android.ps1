param(
    [string]$Serial = 'A3TE025B03003242'
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Adb = 'C:\Users\Administrator\AppData\Local\Android\Sdk\platform-tools\adb.exe'
$Executable = Join-Path $Root 'build\mnn-3.6-android-arm64-opencl\CosyVoiceHiFTBenchmark.out'
$ModelDir = Join-Path $Root 'models\hift-split'
$F0Model = Join-Path $ModelDir 'hift-f0.fp32.mnn'
$CoreModel = Join-Path $ModelDir 'hift-core.fp32.mnn'
$ResultDir = Join-Path $Root 'results\android-hift'
$Remote = '/data/local/tmp/cosy-mnn-stage2'
$Report = Get-Content -LiteralPath (Join-Path $ModelDir 'report.json') -Raw | ConvertFrom-Json
$MelFrames = [int]$Report.mel_shape[2]
foreach ($Required in @($Adb, $Executable, $F0Model, $CoreModel)) {
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
Invoke-Adb push $Executable "$Remote/CosyVoiceHiFTBenchmark.out"
Invoke-Adb push $F0Model "$Remote/hift-f0.mnn"
Invoke-Adb push $CoreModel "$Remote/hift-core.mnn"
Invoke-Adb push (Join-Path $ModelDir 'mel.bin') "$Remote/inputs/mel.bin"
Invoke-Adb push (Join-Path $ModelDir 'source-stft.bin') "$Remote/inputs/source-stft.bin"
Invoke-Adb shell "chmod 755 $Remote/CosyVoiceHiFTBenchmark.out"

$Runs = @(
    @{ Name = 'f0-cpu-high'; Model = 'hift-f0.mnn'; Mode = 'f0'; Backend = 'cpu'; Loops = 1; Threads = 6 },
    @{ Name = 'f0-opencl-high'; Model = 'hift-f0.mnn'; Mode = 'f0'; Backend = 'opencl'; Loops = 3; Threads = 4 },
    @{ Name = 'core-cpu-high'; Model = 'hift-core.mnn'; Mode = 'core'; Backend = 'cpu'; Loops = 1; Threads = 6 },
    @{ Name = 'core-opencl-high'; Model = 'hift-core.mnn'; Mode = 'core'; Backend = 'opencl'; Loops = 3; Threads = 4 }
)
foreach ($Run in $Runs) {
    $RemoteOutput = "$Remote/$($Run.Name).bin"
    & $Adb -s $Serial shell "cd $Remote && ./CosyVoiceHiFTBenchmark.out $($Run.Model) $($Run.Mode) inputs $RemoteOutput $MelFrames $($Run.Backend) high $($Run.Loops) $($Run.Threads)" |
        Tee-Object -FilePath (Join-Path $ResultDir "$($Run.Name).log")
    if ($LASTEXITCODE -ne 0) {
        throw "Android HiFT run failed: $($Run.Name)"
    }
    Invoke-Adb pull $RemoteOutput (Join-Path $ResultDir "$($Run.Name).bin")
    Invoke-Adb pull "$RemoteOutput.json" (Join-Path $ResultDir "$($Run.Name).json")
}

$Python = Join-Path $Root '.venv-export\Scripts\python.exe'
foreach ($Run in $Runs) {
    $Reference = if ($Run.Mode -eq 'f0') { 'f0-pytorch.bin' } else { 'spectral-pytorch.bin' }
    $Tolerance = if ($Run.Mode -eq 'f0') { 0.05 } else { 0.02 }
    & $Python (Join-Path $Root 'scripts\flow_reference.py') compare `
        --reference (Join-Path $ModelDir $Reference) `
        --candidate (Join-Path $ResultDir "$($Run.Name).bin") `
        --report (Join-Path $ResultDir "$($Run.Name)-compare.json") `
        --abs-tolerance $Tolerance
    if ($LASTEXITCODE -ne 0) {
        throw "Android HiFT numerical gate failed: $($Run.Name)"
    }
}

& $Python (Join-Path $Root 'scripts\reconstruct_hift_output.py') `
    --spectral (Join-Path $ResultDir 'core-opencl-high.bin') `
    --mel-frames $MelFrames `
    --reference-pcm (Join-Path $ModelDir 'pcm-pytorch.bin') `
    --output-wav (Join-Path $ResultDir 'hift-opencl-high.wav') `
    --report (Join-Path $ResultDir 'hift-opencl-high-pcm.json')
if ($LASTEXITCODE -ne 0) {
    throw 'Android HiFT OpenCL PCM gate failed'
}
