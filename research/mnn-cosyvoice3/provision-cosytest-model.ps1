param(
    [string]$Adb = 'C:\Users\Administrator\AppData\Local\Android\Sdk\platform-tools\adb.exe',
    [string]$Package = 'io.legado.app.cosytest.debug'
)

$ErrorActionPreference = 'Stop'
$ResearchRoot = $PSScriptRoot
$LlmRoot = 'E:\AndroidStudioProjects\cosyvoice3-distill-lab\Fun-CosyVoice3-0.5B-2512-RL\mnn-llm-int4-speech-eos'
$Files = @(
    @{ Source = Join-Path $LlmRoot 'config-cpu-cosyvoice-ras.json'; Name = 'config-cpu-cosyvoice-ras.json' },
    @{ Source = Join-Path $LlmRoot 'llm_config.json'; Name = 'llm_config.json' },
    @{ Source = Join-Path $LlmRoot 'llm.mnn'; Name = 'llm.mnn' },
    @{ Source = Join-Path $LlmRoot 'llm.mnn.weight'; Name = 'llm.mnn.weight' },
    @{ Source = Join-Path $LlmRoot 'embeddings_bf16.bin'; Name = 'embeddings_bf16.bin' },
    @{ Source = Join-Path $LlmRoot 'tokenizer.mtok'; Name = 'tokenizer.mtok' },
    @{ Source = Join-Path $ResearchRoot 'models\flow-cfg-student-2step-fp16\flow.cfg-student-2step.batch1.fp16.mnn'; Name = 'flow.cfg-student-2step.batch1.fp16.mnn' },
    @{ Source = Join-Path $ResearchRoot 'models\flow-cfg-student-2step-fp16\flow.cfg-student-2step.batch1.fp16.mnn.weight'; Name = 'flow.cfg-student-2step.batch1.fp16.mnn.weight' },
    @{ Source = Join-Path $ResearchRoot 'models\phone-e2e-text-case\conditioner\flow-conditioner.fp32.mnn'; Name = 'flow-conditioner.fp32.mnn' },
    @{ Source = Join-Path $ResearchRoot 'models\phone-e2e-text-case\prompt-speech-tokens.csv'; Name = 'prompt-speech-tokens.csv' },
    @{ Source = Join-Path $ResearchRoot 'models\phone-e2e-text-case\conditioner\prompt-cond.bin'; Name = 'prompt-cond.bin' },
    @{ Source = Join-Path $ResearchRoot 'models\phone-e2e-text-case\conditioner\spks.bin'; Name = 'spks.bin' },
    @{ Source = Join-Path $ResearchRoot 'models\phone-e2e-text-case\conditioner\rand-noise.bin'; Name = 'rand-noise.bin' },
    @{ Source = Join-Path $ResearchRoot 'models\hift-split\hift-f0.fp32.mnn'; Name = 'hift-f0.fp32.mnn' },
    @{ Source = Join-Path $ResearchRoot 'models\hift-split\hift-core.fp32.mnn'; Name = 'hift-core.fp32.mnn' },
    @{ Source = Join-Path $ResearchRoot 'models\phone-e2e-text-case\hift-reference\source-linear-weight.bin'; Name = 'source-linear-weight.bin' },
    @{ Source = Join-Path $ResearchRoot 'models\phone-e2e-text-case\hift-reference\source-linear-bias.bin'; Name = 'source-linear-bias.bin' }
)

if (-not (Test-Path -LiteralPath $Adb)) {
    throw "ADB not found: $Adb"
}
if (-not ((& $Adb devices) -match "`tdevice")) {
    throw 'No ADB device is connected.'
}
if (-not ((& $Adb shell pm path $Package) -match '^package:')) {
    throw "Package is not installed: $Package"
}
$Files | ForEach-Object {
    if (-not (Test-Path -LiteralPath $_.Source)) {
        throw "Model file not found: $($_.Source)"
    }
}

& $Adb shell run-as $Package mkdir -p files/cosyvoice3-mnn/model
if ($LASTEXITCODE -ne 0) { throw 'Cannot create app model directory.' }

$ExpectedTotal = ($Files | ForEach-Object { (Get-Item -LiteralPath $_.Source).Length } | Measure-Object -Sum).Sum
$Completed = 0L
foreach ($File in $Files) {
    $Source = $File.Source
    $Name = $File.Name
    $Length = (Get-Item -LiteralPath $Source).Length
    $Remote = "/data/local/tmp/cosytest-$Name"
    $Percent = [math]::Floor($Completed * 100.0 / $ExpectedTotal)
    Write-Host "[$Percent%] $Name"
    & $Adb push $Source $Remote
    if ($LASTEXITCODE -ne 0) { throw "ADB push failed: $Name" }
    & $Adb shell run-as $Package cp $Remote "files/cosyvoice3-mnn/model/$Name"
    if ($LASTEXITCODE -ne 0) { throw "App copy failed: $Name" }
    $Actual = ((& $Adb shell run-as $Package stat -c '%s' "files/cosyvoice3-mnn/model/$Name") -join '').Trim()
    if ([int64]$Actual -ne $Length) {
        throw "Size mismatch for $Name`: $Actual / $Length"
    }
    & $Adb shell rm -f $Remote
    if ($LASTEXITCODE -ne 0) { throw "Cannot remove transfer file: $Remote" }
    $Completed += $Length
}

Write-Host "[100%] Installed $Completed bytes into $Package"
& $Adb shell run-as $Package ls -l files/cosyvoice3-mnn/model
