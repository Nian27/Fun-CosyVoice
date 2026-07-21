param(
    [switch]$SkipPreparation,
    [ValidateRange(0, 900)][int]$CooldownSeconds = 0,
    [ValidateSet('cpu', 'opencl')][string]$HiFTCoreBackend = 'cpu',
    [ValidateSet('normal', 'low', 'high')][string]$HiFTCorePrecision = 'high',
    [ValidateSet(4, 68, 132)][int]$FlowGpuMode = 4,
    [ValidateSet(4, 68, 132)][int]$HiFTCoreGpuMode = 4,
    [string]$Adb = 'C:\Users\Administrator\AppData\Local\Android\Sdk\platform-tools\adb.exe',
    [string]$RemoteRoot = '/data/local/tmp/cosy-mnn-concurrent-auto',
    [string]$PreparedRoot = '/data/local/tmp/cosy-mnn-persistent-batch',
    [string]$LlmRoot = '/data/local/tmp/cosy-mnn-llm',
    [string]$FlowRoot = '/data/local/tmp/cosy-mnn-cfg-student-2step',
    [string]$HiFTRoot = '/data/local/tmp/cosy-mnn-stage2'
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Build = Join-Path $Root 'build\mnn-3.6-android-arm64-opencl'
$PromptPrefix = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呀。'
$Texts = @(
    '今天天气很好，我们开始测试手机端语音合成速度。',
    '窗外的风停了，房间里只剩下翻书的声音。',
    '下一步要验证连续合成时模型是否会降速或残留上一句状态。'
)
$PromptTokens = 87
$PromptFrames = 174
$GpuCacheRoot = '/data/local/tmp/cosy-mnn-gpu-cache'

function Invoke-Adb {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Adb @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Wait-RemoteFiles {
    param(
        [string[]]$Paths,
        [int]$TimeoutSeconds,
        [string]$Description
    )
    $Watch = [System.Diagnostics.Stopwatch]::StartNew()
    while ($Watch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $Missing = @($Paths | Where-Object {
            ((& $Adb shell "test -f $_; echo `$?") -join '').Trim() -ne '0'
        })
        if ($Missing.Count -eq 0) {
            return
        }
        Start-Sleep -Milliseconds 100
    }
    throw "Timed out waiting for $Description`: $($Paths -join ', ')"
}

function Get-ThermalSnapshot {
    $Raw = ((& $Adb shell dumpsys thermalservice) -join "`n")
    $StatusMatch = [regex]::Match($Raw, 'Thermal Status:\s*(\d+)')
    $HalSection = if ($Raw.Contains('Current temperatures from HAL:')) {
        ($Raw.Split(@('Current temperatures from HAL:'), 2, [System.StringSplitOptions]::None)[1]).Split(
            @('Current cooling devices from HAL:'),
            2,
            [System.StringSplitOptions]::None
        )[0]
    } else {
        ''
    }
    function Get-MaxTemperature([string]$NamePattern) {
        $Matches = [regex]::Matches(
            $HalSection,
            "mValue=([0-9.]+).*mName=$NamePattern[^,}]*"
        )
        if ($Matches.Count -eq 0) { return $null }
        return ($Matches | ForEach-Object { [double]$_.Groups[1].Value } |
            Measure-Object -Maximum).Maximum
    }
    [ordered]@{
        status = if ($StatusMatch.Success) { [int]$StatusMatch.Groups[1].Value } else { $null }
        cpuMaxC = Get-MaxTemperature 'CPU[0-9]+'
        gpuMaxC = Get-MaxTemperature 'GPU[0-9]+'
        skinC = Get-MaxTemperature 'skin'
        batteryC = Get-MaxTemperature 'battery'
    }
}

foreach ($Required in @(
    $Adb,
    (Join-Path $Build 'CosyVoiceLlmPersistentBenchmark.out'),
    (Join-Path $Build 'CosyVoiceFlowPersistentBenchmark.out'),
    (Join-Path $Build 'CosyVoiceHiFTPersistentBenchmark.out')
)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required file not found: $Required"
    }
}

if (-not ((& $Adb devices) -match "`tdevice")) {
    throw 'No ADB device is connected.'
}

if (-not $SkipPreparation) {
    & (Join-Path $Root 'run-phone-persistent-batch.ps1') -Adb $Adb
    if ($LASTEXITCODE -ne 0) {
        throw 'Persistent batch preparation failed.'
    }
}

if ($CooldownSeconds -gt 0) {
    Write-Host "Cooling down for $CooldownSeconds seconds before the synchronized run."
    Start-Sleep -Seconds $CooldownSeconds
}

$RemoteRequirements = @(
    "$LlmRoot/config-cpu-cosyvoice-ras.json",
    "$FlowRoot/flow.mnn",
    "$HiFTRoot/hift-f0.mnn",
    "$HiFTRoot/hift-core.mnn"
)
for ($Index = 0; $Index -lt $Texts.Count; $Index++) {
    $RemoteRequirements += "$PreparedRoot/llm-output/speech-tokens-$Index.csv"
    $RemoteRequirements += "$PreparedRoot/request-$Index/flow-input/x.bin"
    $RemoteRequirements += "$PreparedRoot/request-$Index/hift-input/mel.bin"
}
Wait-RemoteFiles $RemoteRequirements 10 'models and prepared inputs'

$Timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$Result = Join-Path $Root "results\phone-concurrent\$Timestamp"
$PromptFile = Join-Path $Result 'prompts.txt'
$FlowManifest = Join-Path $Result 'flow-manifest.txt'
$HiFTManifest = Join-Path $Result 'hift-manifest.txt'
New-Item -ItemType Directory -Force $Result | Out-Null
$Texts | ForEach-Object { $PromptPrefix + $_ } |
    Set-Content -LiteralPath $PromptFile -Encoding utf8

$Requests = @()
$FlowLines = @()
$HiFTLines = @()
for ($Index = 0; $Index -lt $Texts.Count; $Index++) {
    $TokenCsv = ((& $Adb shell "cat $PreparedRoot/llm-output/speech-tokens-$Index.csv") -join '').Trim()
    $TargetTokens = @($TokenCsv.Split(',', [System.StringSplitOptions]::RemoveEmptyEntries)).Count
    if ($TargetTokens -lt 1) {
        throw "Prepared request $Index has no speech tokens."
    }
    $SequenceLength = 2 * ($PromptTokens + $TargetTokens)
    if ($SequenceLength -gt 512) {
        throw "Prepared request $Index needs sequence $SequenceLength; this benchmark currently fixes Flow to bucket 512."
    }
    $TargetFrames = 2 * $TargetTokens
    $FlowInput = "$PreparedRoot/request-$Index/flow-input"
    $HiFTInput = "$PreparedRoot/request-$Index/hift-input"
    $FlowLines += "$FlowInput $RemoteRoot/flow-out-$Index $SequenceLength $PromptFrames 512"
    $HiFTLines += "$HiFTInput $RemoteRoot/hift-out-$Index $TargetFrames"
    $Requests += [pscustomobject]@{
        index = $Index
        targetTokens = $TargetTokens
        targetFrames = $TargetFrames
        audioSeconds = $TargetFrames * 0.02
    }
}

# The first Flow request compiles and warms bucket 512 before the common start barrier.
$FlowLines = @($FlowLines[0]) + $FlowLines
$FlowLines | Set-Content -LiteralPath $FlowManifest -Encoding ascii
$HiFTLines | Set-Content -LiteralPath $HiFTManifest -Encoding ascii

Invoke-Adb shell "rm -rf $RemoteRoot; mkdir -p $RemoteRoot/llm-output"
Invoke-Adb shell "mkdir -p $GpuCacheRoot"
Invoke-Adb push (Join-Path $Build 'CosyVoiceLlmPersistentBenchmark.out') "$RemoteRoot/llm"
Invoke-Adb push (Join-Path $Build 'CosyVoiceFlowPersistentBenchmark.out') "$RemoteRoot/flow"
Invoke-Adb push (Join-Path $Build 'CosyVoiceHiFTPersistentBenchmark.out') "$RemoteRoot/hift"
Invoke-Adb push $PromptFile "$RemoteRoot/prompts.txt"
Invoke-Adb push $FlowManifest "$RemoteRoot/flow-manifest.txt"
Invoke-Adb push $HiFTManifest "$RemoteRoot/hift-manifest.txt"
Invoke-Adb shell "chmod 755 $RemoteRoot/llm $RemoteRoot/flow $RemoteRoot/hift; mkdir -p $RemoteRoot/flow-out-0 $RemoteRoot/flow-out-1 $RemoteRoot/flow-out-2 $RemoteRoot/hift-out-0 $RemoteRoot/hift-out-1 $RemoteRoot/hift-out-2"

$LlmCommand = "cd $LlmRoot; $RemoteRoot/llm config-cpu-cosyvoice-ras.json $RemoteRoot/prompts.txt 500 $PreparedRoot/voice/prompt-speech-tokens.csv $RemoteRoot/llm-output $RemoteRoot/llm.ready $RemoteRoot/start > $RemoteRoot/llm.log 2>&1; echo `$? > $RemoteRoot/llm.status"
$FlowCache = "$GpuCacheRoot/flow-fp32-opencl-high-mode$FlowGpuMode.cache"
$HiFTCache = "$GpuCacheRoot/hift-core-opencl-$HiFTCorePrecision-mode$HiFTCoreGpuMode.cache"
$FlowCommand = "cd $RemoteRoot; ./flow $FlowRoot/flow.mnn flow-manifest.txt opencl high $FlowGpuMode flow-report.jsonl flow.ready start 1 $FlowCache > flow.log 2>&1; echo `$? > flow.status"
$HiFTCommand = "cd $RemoteRoot; ./hift $HiFTRoot/hift-f0.mnn $HiFTRoot/hift-core.mnn hift-manifest.txt 4 hift-report.jsonl hift.ready start $HiFTCoreBackend $HiFTCorePrecision $HiFTCache $HiFTCoreGpuMode > hift.log 2>&1; echo `$? > hift.status"
Invoke-Adb shell "nohup sh -c '$LlmCommand' >/dev/null 2>&1 &"
Invoke-Adb shell "nohup sh -c '$FlowCommand' >/dev/null 2>&1 &"
Invoke-Adb shell "nohup sh -c '$HiFTCommand' >/dev/null 2>&1 &"

Wait-RemoteFiles @("$RemoteRoot/llm.ready", "$RemoteRoot/flow.ready", "$RemoteRoot/hift.ready") 90 'benchmark workers to become ready'
$ThermalBefore = Get-ThermalSnapshot
$ConcurrentWatch = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-Adb shell "touch $RemoteRoot/start"
Wait-RemoteFiles @("$RemoteRoot/llm.status", "$RemoteRoot/flow.status", "$RemoteRoot/hift.status") 120 'benchmark workers to finish'
$ConcurrentWatch.Stop()
$ThermalAfter = Get-ThermalSnapshot

$Statuses = [ordered]@{}
foreach ($Worker in @('llm', 'flow', 'hift')) {
    $Statuses[$Worker] = [int](((& $Adb shell "cat $RemoteRoot/$Worker.status") -join '').Trim())
    Invoke-Adb pull "$RemoteRoot/$Worker.log" (Join-Path $Result "$Worker.log")
}
Invoke-Adb pull "$RemoteRoot/llm-output/llm-persistent.jsonl" (Join-Path $Result 'llm-persistent.jsonl')
Invoke-Adb pull "$RemoteRoot/flow-report.jsonl" (Join-Path $Result 'flow-report.jsonl')
Invoke-Adb pull "$RemoteRoot/hift-report.jsonl" (Join-Path $Result 'hift-report.jsonl')

if (@($Statuses.Values | Where-Object { $_ -ne 0 }).Count -gt 0) {
    throw "Concurrent worker failed: $($Statuses | ConvertTo-Json -Compress)"
}

$LlmMetrics = @(Get-Content (Join-Path $Result 'llm-persistent.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
$FlowMetrics = @(Get-Content (Join-Path $Result 'flow-report.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
$HiFTMetrics = @(Get-Content (Join-Path $Result 'hift-report.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
$MeasuredFlow = @($FlowMetrics | Select-Object -Skip 1)
$TotalAudioSeconds = ($Requests | Measure-Object -Property audioSeconds -Sum).Sum
$Summary = [ordered]@{
    requestCount = $Requests.Count
    totalAudioSeconds = $TotalAudioSeconds
    concurrentWallSeconds = $ConcurrentWatch.Elapsed.TotalSeconds
    throughputRtf = $ConcurrentWatch.Elapsed.TotalSeconds / $TotalAudioSeconds
    statuses = $Statuses
    llmWallSeconds = ($LlmMetrics | Measure-Object -Property wallMs -Sum).Sum / 1000
    flowInferenceSeconds = ($MeasuredFlow | Measure-Object -Property inferenceMs -Sum).Sum / 1000
    flowGpuMode = $FlowGpuMode
    hiftSeconds = ($HiFTMetrics | Measure-Object -Property requestMs -Sum).Sum / 1000
    hiftCoreBackend = $HiFTCoreBackend
    hiftCorePrecision = $HiFTCorePrecision
    hiftCoreGpuMode = $HiFTCoreGpuMode
    flowWarmupResizeSeconds = $FlowMetrics[0].resizeMs / 1000
    pcmPeaks = @($HiFTMetrics | ForEach-Object { $_.pcmPeak })
    pcmRms = @($HiFTMetrics | ForEach-Object { $_.pcmRms })
    cooldownSeconds = $CooldownSeconds
    thermalBefore = $ThermalBefore
    thermalAfter = $ThermalAfter
    resultDirectory = $Result
}
$Summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Result 'summary.json') -Encoding utf8
$Summary | ConvertTo-Json -Depth 5
