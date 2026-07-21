param(
    [string[]]$Texts = @(
        '今天天气很好，我们开始测试手机端语音合成速度。',
        '窗外的风停了，房间里只剩下翻书的声音。',
        '下一步要验证连续合成时模型是否会降速或残留上一句状态。'
    ),
    [string]$Adb = 'C:\Users\Administrator\AppData\Local\Android\Sdk\platform-tools\adb.exe',
    [string]$RemoteRoot = '/data/local/tmp/cosy-mnn-persistent-batch',
    [string]$LlmRoot = '/data/local/tmp/cosy-mnn-llm',
    [string]$FlowRoot = '/data/local/tmp/cosy-mnn-cfg-student-2step',
    [string]$HiFTRoot = '/data/local/tmp/cosy-mnn-stage2'
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Build = Join-Path $Root 'build\mnn-3.6-android-arm64-opencl'
$Assets = Join-Path $Root 'models\phone-e2e-text-case'
$ConditionerAssets = Join-Path $Assets 'conditioner'
$HiFTAssets = Join-Path $Assets 'hift-reference'
$PromptPrefix = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呀。'
$PromptTokens = 87
$PromptFrames = 174

function Invoke-Adb {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Adb @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Get-FlowBucket([int]$SequenceLength) {
    foreach ($Bucket in @(512, 768, 1024, 1280, 1536, 2048)) {
        if ($SequenceLength -le $Bucket) {
            return $Bucket
        }
    }
    throw "Sequence length $SequenceLength exceeds the supported Flow buckets."
}

if ($Texts.Count -lt 1) {
    throw 'At least one text is required.'
}

foreach ($Required in @(
    $Adb,
    (Join-Path $Build 'CosyVoiceLlmPersistentBenchmark.out'),
    (Join-Path $Build 'CosyVoiceFlowConditionerBenchmark.out'),
    (Join-Path $Build 'CosyVoiceFlowPersistentBenchmark.out'),
    (Join-Path $Build 'CosyVoiceHiFTPersistentBenchmark.out'),
    (Join-Path $ConditionerAssets 'flow-conditioner.fp32.mnn'),
    (Join-Path $Assets 'prompt-speech-tokens.csv'),
    (Join-Path $ConditionerAssets 'prompt-cond.bin'),
    (Join-Path $ConditionerAssets 'spks.bin'),
    (Join-Path $ConditionerAssets 'rand-noise.bin'),
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
$Result = Join-Path $Root "results\phone-persistent-batch\$Timestamp"
$PromptFile = Join-Path $Result 'prompts.txt'
$ManifestFile = Join-Path $Result 'flow-manifest.txt'
New-Item -ItemType Directory -Force $Result | Out-Null
$Texts | ForEach-Object { $PromptPrefix + $_ } |
    Set-Content -LiteralPath $PromptFile -Encoding utf8

Invoke-Adb shell "rm -rf $RemoteRoot; mkdir -p $RemoteRoot/voice $RemoteRoot/llm-output"
Invoke-Adb push (Join-Path $Build 'CosyVoiceLlmPersistentBenchmark.out') "$RemoteRoot/llm-persistent"
Invoke-Adb push (Join-Path $Build 'CosyVoiceFlowConditionerBenchmark.out') "$RemoteRoot/conditioner"
Invoke-Adb push (Join-Path $Build 'CosyVoiceFlowPersistentBenchmark.out') "$RemoteRoot/flow-persistent"
Invoke-Adb push (Join-Path $Build 'CosyVoiceHiFTPersistentBenchmark.out') "$RemoteRoot/hift-persistent"
Invoke-Adb push (Join-Path $ConditionerAssets 'flow-conditioner.fp32.mnn') "$RemoteRoot/conditioner.mnn"
Invoke-Adb push (Join-Path $Assets 'prompt-speech-tokens.csv') "$RemoteRoot/voice/prompt-speech-tokens.csv"
Invoke-Adb push (Join-Path $ConditionerAssets 'prompt-cond.bin') "$RemoteRoot/voice/prompt-cond.bin"
Invoke-Adb push (Join-Path $ConditionerAssets 'spks.bin') "$RemoteRoot/voice/spks.bin"
Invoke-Adb push (Join-Path $ConditionerAssets 'rand-noise.bin') "$RemoteRoot/voice/rand-noise.bin"
Invoke-Adb push (Join-Path $HiFTAssets 'source-linear-weight.bin') "$RemoteRoot/voice/source-linear-weight.bin"
Invoke-Adb push (Join-Path $HiFTAssets 'source-linear-bias.bin') "$RemoteRoot/voice/source-linear-bias.bin"
Invoke-Adb push $PromptFile "$RemoteRoot/prompts.txt"
Invoke-Adb shell "chmod 755 $RemoteRoot/llm-persistent $RemoteRoot/conditioner $RemoteRoot/flow-persistent $RemoteRoot/hift-persistent"

$PipelineWatch = [System.Diagnostics.Stopwatch]::StartNew()
$LlmOutput = & $Adb shell "cd $LlmRoot && $RemoteRoot/llm-persistent config-cpu-cosyvoice-ras.json $RemoteRoot/prompts.txt 500 $RemoteRoot/voice/prompt-speech-tokens.csv $RemoteRoot/llm-output"
if ($LASTEXITCODE -ne 0) {
    throw "Persistent LLM failed: $($LlmOutput -join "`n")"
}
$LlmOutput | Set-Content -LiteralPath (Join-Path $Result 'llm-console.log') -Encoding utf8

$Requests = @()
$ManifestLines = @()
for ($Index = 0; $Index -lt $Texts.Count; $Index++) {
    $TokenFile = "$RemoteRoot/llm-output/speech-tokens-$Index.csv"
    $TokenCsv = ((& $Adb shell "cat $TokenFile") -join '').Trim()
    $TargetTokens = @($TokenCsv.Split(',', [System.StringSplitOptions]::RemoveEmptyEntries)).Count
    if ($TargetTokens -lt 1) {
        throw "LLM request $Index produced no speech tokens."
    }
    $SequenceLength = 2 * ($PromptTokens + $TargetTokens)
    $TargetFrames = 2 * $TargetTokens
    $Bucket = Get-FlowBucket $SequenceLength
    $FlowInput = "$RemoteRoot/request-$Index/flow-input"
    $FlowOutput = "$RemoteRoot/request-$Index/flow-output"
    $HiFTInput = "$RemoteRoot/request-$Index/hift-input"
    $HiFTOutput = "$RemoteRoot/request-$Index/hift-output"
    Invoke-Adb shell "mkdir -p $FlowInput $FlowOutput $HiFTInput $HiFTOutput"
    Invoke-Adb shell "cd $RemoteRoot && ./conditioner conditioner.mnn voice $TokenFile $FlowInput $PromptTokens $PromptFrames"
    $ManifestLines += "$FlowInput $FlowOutput $SequenceLength $PromptFrames $Bucket"
    $Requests += [pscustomobject]@{
        index = $Index
        text = $Texts[$Index]
        targetTokens = $TargetTokens
        sequenceLength = $SequenceLength
        targetFrames = $TargetFrames
        flowBucket = $Bucket
        audioSeconds = $TargetFrames * 0.02
        hiftInput = $HiFTInput
        hiftOutput = $HiFTOutput
        flowOutput = $FlowOutput
    }
}

$ManifestLines | Set-Content -LiteralPath $ManifestFile -Encoding ascii
Invoke-Adb push $ManifestFile "$RemoteRoot/flow-manifest.txt"
$FlowOutput = & $Adb shell "cd $RemoteRoot && ./flow-persistent $FlowRoot/flow.mnn flow-manifest.txt opencl high 4 flow-report.jsonl"
if ($LASTEXITCODE -ne 0) {
    throw "Persistent Flow failed: $($FlowOutput -join "`n")"
}
$FlowOutput | Set-Content -LiteralPath (Join-Path $Result 'flow-console.log') -Encoding utf8

$HiFTManifestFile = Join-Path $Result 'hift-manifest.txt'
$HiFTManifestLines = @()
foreach ($Request in $Requests) {
    Invoke-Adb shell "cp $($Request.flowOutput)/student_target_mel_android.bin $($Request.hiftInput)/mel.bin; cp $RemoteRoot/voice/source-linear-weight.bin $($Request.hiftInput)/; cp $RemoteRoot/voice/source-linear-bias.bin $($Request.hiftInput)/"
    $HiFTManifestLines += "$($Request.hiftInput) $($Request.hiftOutput) $($Request.targetFrames)"
}
$HiFTManifestLines | Set-Content -LiteralPath $HiFTManifestFile -Encoding ascii
Invoke-Adb push $HiFTManifestFile "$RemoteRoot/hift-manifest.txt"
Invoke-Adb shell "cd $RemoteRoot && ./hift-persistent $HiFTRoot/hift-f0.mnn $HiFTRoot/hift-core.mnn hift-manifest.txt 4 hift-report.jsonl"
$PipelineWatch.Stop()

Invoke-Adb pull "$RemoteRoot/llm-output/llm-persistent.jsonl" (Join-Path $Result 'llm-persistent.jsonl')
Invoke-Adb pull "$RemoteRoot/flow-report.jsonl" (Join-Path $Result 'flow-report.jsonl')
Invoke-Adb pull "$RemoteRoot/hift-report.jsonl" (Join-Path $Result 'hift-report.jsonl')
for ($Index = 0; $Index -lt $Requests.Count; $Index++) {
    $RequestResult = Join-Path $Result "request-$Index"
    New-Item -ItemType Directory -Force $RequestResult | Out-Null
    Invoke-Adb pull "$RemoteRoot/request-$Index/hift-output/hift-android.wav" (Join-Path $RequestResult 'output.wav')
}

$LlmMetrics = @(Get-Content (Join-Path $Result 'llm-persistent.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
$FlowMetrics = @(Get-Content (Join-Path $Result 'flow-report.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
$HiFTMetrics = @(Get-Content (Join-Path $Result 'hift-report.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
$TotalAudioSeconds = 0.0
$LlmWallSeconds = 0.0
$FlowInferenceSeconds = 0.0
$FlowResizeSeconds = 0.0
$HiFTSeconds = 0.0
$Details = @()
for ($Index = 0; $Index -lt $Requests.Count; $Index++) {
    $HiFT = $HiFTMetrics[$Index]
    $Request = $Requests[$Index]
    $TotalAudioSeconds += $Request.audioSeconds
    $LlmWallSeconds += $LlmMetrics[$Index].wallMs / 1000
    $FlowInferenceSeconds += $FlowMetrics[$Index].inferenceMs / 1000
    $FlowResizeSeconds += $FlowMetrics[$Index].resizeMs / 1000
    $HiFTSeconds += $HiFT.requestMs / 1000
    $Details += [ordered]@{
        index = $Index
        text = $Request.text
        targetTokens = $Request.targetTokens
        audioSeconds = $Request.audioSeconds
        flowBucket = $Request.flowBucket
        llmWallSeconds = $LlmMetrics[$Index].wallMs / 1000
        flowResizeSeconds = $FlowMetrics[$Index].resizeMs / 1000
        flowInferenceSeconds = $FlowMetrics[$Index].inferenceMs / 1000
        hiftResizeSeconds = $HiFT.resizeMs / 1000
        hiftSeconds = $HiFT.requestMs / 1000
        pcmPeak = $HiFT.pcmPeak
        pcmRms = $HiFT.pcmRms
    }
}

$Summary = [ordered]@{
    requestCount = $Requests.Count
    totalAudioSeconds = $TotalAudioSeconds
    measuredColdBatchSeconds = $PipelineWatch.Elapsed.TotalSeconds
    measuredColdBatchRtf = $PipelineWatch.Elapsed.TotalSeconds / $TotalAudioSeconds
    llmRequestSeconds = $LlmWallSeconds
    flowFirstShapeCompileSeconds = $FlowResizeSeconds
    flowInferenceSeconds = $FlowInferenceSeconds
    hiftSeconds = $HiFTSeconds
    nonOverlappedWarmWorkSeconds = $LlmWallSeconds + $FlowInferenceSeconds + $HiFTSeconds
    nonOverlappedWarmWorkRtf = ($LlmWallSeconds + $FlowInferenceSeconds + $HiFTSeconds) / $TotalAudioSeconds
    details = $Details
    resultDirectory = $Result
}
$Summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Result 'summary.json') -Encoding utf8
$Summary | ConvertTo-Json -Depth 6
