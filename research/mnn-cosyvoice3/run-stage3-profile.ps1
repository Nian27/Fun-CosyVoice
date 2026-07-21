param(
    [string]$Python = 'D:\anaconda3\envs\pytorch\python.exe',
    [string]$Model = 'E:\AndroidStudioProjects\cosyvoice3-distill-lab\Fun-CosyVoice3-0.5B-2512-RL',
    [string]$Repo = 'E:\AndroidStudioProjects\CosyVoice-main',
    [string]$DependencyPath = 'E:\AndroidStudioProjects\cosyvoice3-distill-lab\python-deps',
    [int]$Steps = 10,
    [int]$Repeats = 3
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = $DependencyPath
$Output = Join-Path $PSScriptRoot "results\stage3-pipeline-profile-$Steps"
& $Python (Join-Path $PSScriptRoot 'scripts\profile_cosyvoice_pipeline.py') `
    --repo $Repo `
    --model $Model `
    --prompt-wav (Join-Path $Repo 'asset\zero_shot_prompt.wav') `
    --output $Output `
    --steps $Steps `
    --repeats $Repeats
if ($LASTEXITCODE -ne 0) {
    throw 'Stage 3 pipeline profile failed'
}
