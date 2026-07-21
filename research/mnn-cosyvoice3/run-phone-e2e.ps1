param(
    [string]$Text = '今天天气很好，我们开始测试手机端语音合成速度。',
    [string]$Adb = 'C:\Users\Administrator\AppData\Local\Android\Sdk\platform-tools\adb.exe',
    [string]$RemoteRoot = '/data/local/tmp/cosy-mnn-e2e-live',
    [string]$LlmRoot = '/data/local/tmp/cosy-mnn-llm',
    [string]$FlowRoot = '/data/local/tmp/cosy-mnn-cfg-student-2step',
    [string]$HiFTRoot = '/data/local/tmp/cosy-mnn-stage2'
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Build = Join-Path $Root 'build\mnn-3.6-android-arm64-opencl'
$ConditionerAssets = Join-Path $Root 'models\phone-e2e-text-case\conditioner'
$VoiceAssets = Join-Path $Root 'models\phone-e2e-text-case'
$HiFTAssets = Join-Path $Root 'models\phone-e2e-text-case\hift-reference'
$PromptPrefix = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呀。'

function Invoke-Adb {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Adb @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

foreach ($Required in @(
    $Adb,
    (Join-Path $Build 'CosyVoiceLlmBenchmark.out'),
    (Join-Path $Build 'CosyVoiceFlowConditionerBenchmark.out'),
    (Join-Path $Build 'CosyVoiceTwoStepFlowBenchmark.out'),
    (Join-Path $Build 'CosyVoiceHiFTE2EBenchmark.out'),
    (Join-Path $ConditionerAssets 'flow-conditioner.fp32.mnn'),
    (Join-Path $ConditionerAssets 'prompt-cond.bin'),
    (Join-Path $ConditionerAssets 'spks.bin'),
    (Join-Path $ConditionerAssets 'rand-noise.bin'),
    (Join-Path $VoiceAssets 'prompt-speech-tokens.csv'),
    (Join-Path $HiFTAssets 'source-linear-weight.bin'),
    (Join-Path $HiFTAssets 'source-linear-bias.bin')
)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required file not found: $Required"
    }
}

$Devices = & $Adb devices
if (-not ($Devices -match "`tdevice")) {
    throw 'No ADB device is connected.'
}

$Timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$Result = Join-Path $Root "results\phone-e2e\$Timestamp"
$TempText = Join-Path $Result 'input.txt'
New-Item -ItemType Directory -Force $Result | Out-Null
Set-Content -LiteralPath $TempText -Value ($PromptPrefix + $Text) -Encoding utf8 -NoNewline

Invoke-Adb shell "rm -rf $RemoteRoot; mkdir -p $RemoteRoot/voice $RemoteRoot/flow-input $RemoteRoot/flow-output $RemoteRoot/hift-input $RemoteRoot/hift-output"
Invoke-Adb push (Join-Path $Build 'CosyVoiceLlmBenchmark.out') "$LlmRoot/llm_bench"
Invoke-Adb push (Join-Path $Build 'CosyVoiceFlowConditionerBenchmark.out') "$RemoteRoot/conditioner"
Invoke-Adb push (Join-Path $Build 'CosyVoiceTwoStepFlowBenchmark.out') "$RemoteRoot/flow2"
Invoke-Adb push (Join-Path $Build 'CosyVoiceHiFTE2EBenchmark.out') "$RemoteRoot/hift_e2e"
Invoke-Adb push (Join-Path $ConditionerAssets 'flow-conditioner.fp32.mnn') "$RemoteRoot/conditioner.mnn"
Invoke-Adb push (Join-Path $VoiceAssets 'prompt-speech-tokens.csv') "$RemoteRoot/voice/prompt-speech-tokens.csv"
Invoke-Adb push (Join-Path $ConditionerAssets 'prompt-cond.bin') "$RemoteRoot/voice/prompt-cond.bin"
Invoke-Adb push (Join-Path $ConditionerAssets 'spks.bin') "$RemoteRoot/voice/spks.bin"
Invoke-Adb push (Join-Path $ConditionerAssets 'rand-noise.bin') "$RemoteRoot/voice/rand-noise.bin"
Invoke-Adb push (Join-Path $HiFTAssets 'source-linear-weight.bin') "$RemoteRoot/voice/source-linear-weight.bin"
Invoke-Adb push (Join-Path $HiFTAssets 'source-linear-bias.bin') "$RemoteRoot/voice/source-linear-bias.bin"
Invoke-Adb push $TempText "$RemoteRoot/input.txt"
Invoke-Adb shell "chmod 755 $LlmRoot/llm_bench $RemoteRoot/conditioner $RemoteRoot/flow2 $RemoteRoot/hift_e2e"

$PipelineWatch = [System.Diagnostics.Stopwatch]::StartNew()
$LlmWatch = [System.Diagnostics.Stopwatch]::StartNew()
$LlmOutput = & $Adb shell "cd $LlmRoot && ./llm_bench config-cpu-cosyvoice-ras.json @$RemoteRoot/input.txt 500 $RemoteRoot/voice/prompt-speech-tokens.csv $RemoteRoot/output-speech-tokens.csv"
$LlmWatch.Stop()
if ($LASTEXITCODE -ne 0) {
    throw "LLM stage failed: $($LlmOutput -join "`n")"
}
$LlmOutput | Set-Content -LiteralPath (Join-Path $Result 'llm.log') -Encoding utf8
$TokenCsv = ((& $Adb shell "cat $RemoteRoot/output-speech-tokens.csv") -join '').Trim()
$TargetTokens = @($TokenCsv.Split(',', [System.StringSplitOptions]::RemoveEmptyEntries)).Count
if ($TargetTokens -lt 1) {
    throw 'LLM produced no speech tokens.'
}
$PromptTokens = 87
$PromptFrames = 174
$SequenceLength = 2 * ($PromptTokens + $TargetTokens)
$TargetFrames = 2 * $TargetTokens

$ConditionerWatch = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-Adb shell "cd $RemoteRoot && ./conditioner conditioner.mnn voice output-speech-tokens.csv flow-input $PromptTokens $PromptFrames"
$ConditionerWatch.Stop()
$FlowWatch = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-Adb shell "cd $RemoteRoot/flow-output && ../flow2 $FlowRoot/flow.mnn ../flow-input . $SequenceLength $PromptFrames opencl high 1 4"
$FlowWatch.Stop()
Invoke-Adb shell "cp $RemoteRoot/flow-output/student_target_mel_android.bin $RemoteRoot/hift-input/mel.bin; cp $RemoteRoot/voice/source-linear-weight.bin $RemoteRoot/hift-input/; cp $RemoteRoot/voice/source-linear-bias.bin $RemoteRoot/hift-input/"
$HiFTWatch = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-Adb shell "cd $RemoteRoot/hift-output && ../hift_e2e $HiFTRoot/hift-f0.mnn $HiFTRoot/hift-core.mnn ../hift-input . $TargetFrames 4"
$HiFTWatch.Stop()
$PipelineWatch.Stop()
Invoke-Adb shell "cp $RemoteRoot/hift-output/hift-android.wav /sdcard/Download/cosyvoice3-mnn-latest.wav"

Invoke-Adb pull "$RemoteRoot/output-speech-tokens.csv" (Join-Path $Result 'output-speech-tokens.csv')
Invoke-Adb pull "$RemoteRoot/flow-input/conditioner-android.json" (Join-Path $Result 'conditioner-android.json')
Invoke-Adb pull "$RemoteRoot/flow-output/flow_two_step_android.json" (Join-Path $Result 'flow-two-step-android.json')
Invoke-Adb pull "$RemoteRoot/hift-output/hift_e2e_android.json" (Join-Path $Result 'hift-e2e-android.json')
Invoke-Adb pull "$RemoteRoot/hift-output/hift-android.wav" (Join-Path $Result 'output.wav')

$Conditioner = Get-Content -Raw (Join-Path $Result 'conditioner-android.json') | ConvertFrom-Json
$Flow = Get-Content -Raw (Join-Path $Result 'flow-two-step-android.json') | ConvertFrom-Json
$HiFT = Get-Content -Raw (Join-Path $Result 'hift-e2e-android.json') | ConvertFrom-Json
$Prefill = [double](($LlmOutput | Select-String '^prefill_seconds=').Line.Split('=')[1])
$Decode = [double](($LlmOutput | Select-String '^decode_seconds=').Line.Split('=')[1])
$InferenceOnlySeconds = $Prefill + $Decode + $Conditioner.inferenceMs / 1000 + $Flow.twoStepMedianMs / 1000 + $HiFT.totalMs / 1000
$AudioSeconds = $TargetFrames * 0.02
$Summary = [ordered]@{
    text = $Text
    promptTokens = $PromptTokens
    targetTokens = $TargetTokens
    targetFrames = $TargetFrames
    audioSeconds = $AudioSeconds
    llmSeconds = $Prefill + $Decode
    conditionerSeconds = $Conditioner.inferenceMs / 1000
    flowSeconds = $Flow.twoStepMedianMs / 1000
    hiftSeconds = $HiFT.totalMs / 1000
    inferenceOnlySeconds = $InferenceOnlySeconds
    inferenceOnlyRtf = $InferenceOnlySeconds / $AudioSeconds
    llmWallSeconds = $LlmWatch.Elapsed.TotalSeconds
    conditionerWallSeconds = $ConditionerWatch.Elapsed.TotalSeconds
    flowWallSeconds = $FlowWatch.Elapsed.TotalSeconds
    hiftWallSeconds = $HiFTWatch.Elapsed.TotalSeconds
    coldPipelineWallSeconds = $PipelineWatch.Elapsed.TotalSeconds
    coldPipelineRtf = $PipelineWatch.Elapsed.TotalSeconds / $AudioSeconds
    flowLoadAndResizeSeconds = $Flow.loadMs / 1000
    pcmPeak = $HiFT.pcmPeak
    pcmRms = $HiFT.pcmRms
    phoneWave = '/sdcard/Download/cosyvoice3-mnn-latest.wav'
    resultDirectory = $Result
}
$Summary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Result 'summary.json') -Encoding utf8
$Summary | ConvertTo-Json
