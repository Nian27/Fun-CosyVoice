param(
    [string]$Repo = 'E:\AndroidStudioProjects\CosyVoice-main',
    [string]$Model = 'E:\AndroidStudioProjects\cosyvoice3-distill-lab\Fun-CosyVoice3-0.5B-2512-RL',
    [string]$Python = 'D:\anaconda3\envs\pytorch\python.exe',
    [string]$PythonDependencies = 'E:\AndroidStudioProjects\cosyvoice3-distill-lab\python-deps',
    [int]$CfgSteps = 1000,
    [int]$TwoStepSteps = 2000
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$PromptWav = Join-Path $Repo 'asset\zero_shot_prompt.wav'
$TextFile = Join-Path $Root 'data\cfg_distill_texts_zh.txt'
$ResultDir = Join-Path $Root 'results\cfg-distill-real'
$TeacherCache = Join-Path $ResultDir 'teacher-real-24utt.pt'
$CfgStudent = Join-Path $ResultDir 'student-last8-real-24utt.pt'
$TwoStepCache = Join-Path $ResultDir 'teacher-real-24utt-2step.pt'
$TwoStepStudent = Join-Path $ResultDir 'student-last8-real-24utt-2step.pt'
$EndpointReport = Join-Path $ResultDir 'two-step-endpoint-validation.json'

foreach ($Required in @($Python, $Repo, $Model, $PromptWav, $TextFile)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required path not found: $Required"
    }
}

$env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $PythonDependencies
} else {
    "$PythonDependencies;$env:PYTHONPATH"
}
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python stage failed: $($Arguments -join ' ')"
    }
}

Push-Location $Root
try {
    Invoke-Python @(
        'scripts\capture_cfg_trajectory.py',
        '--repo', $Repo,
        '--model', $Model,
        '--prompt-wav', $PromptWav,
        '--text-file', $TextFile,
        '--output', $TeacherCache,
        '--validation-utterances', '4'
    )
    Invoke-Python @(
        'scripts\train_cfg_student.py',
        '--repo', $Repo,
        '--model', $Model,
        '--cache', $TeacherCache,
        '--output', $CfgStudent,
        '--steps', "$CfgSteps",
        '--last-blocks', '8',
        '--learning-rate', '0.00001'
    )
    Invoke-Python @(
        'scripts\build_two_step_cache.py',
        '--input', $TeacherCache,
        '--output', $TwoStepCache
    )
    Invoke-Python @(
        'scripts\train_cfg_student.py',
        '--repo', $Repo,
        '--model', $Model,
        '--cache', $TwoStepCache,
        '--initial-patch', $CfgStudent,
        '--output', $TwoStepStudent,
        '--steps', "$TwoStepSteps",
        '--last-blocks', '8',
        '--learning-rate', '0.000005'
    )
    Invoke-Python @(
        'scripts\evaluate_two_step_endpoint.py',
        '--repo', $Repo,
        '--model', $Model,
        '--cache', $TeacherCache,
        '--student-patch', $TwoStepStudent,
        '--output', $EndpointReport
    )
} finally {
    Pop-Location
}

Write-Host "Two-step student: $TwoStepStudent"
Write-Host "Endpoint report: $EndpointReport"
