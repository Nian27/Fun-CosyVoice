param(
    [string]$Python = 'D:\anaconda3\envs\pytorch\python.exe',
    [string]$Model = 'E:\AndroidStudioProjects\cosyvoice3-distill-lab\Fun-CosyVoice3-0.5B-2512-RL',
    [string]$Repo = 'E:\AndroidStudioProjects\CosyVoice-main',
    [string]$DependencyPath = 'E:\AndroidStudioProjects\cosyvoice3-distill-lab\python-deps'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = $DependencyPath
$Output = Join-Path $PSScriptRoot 'results\stage3-flow-steps'
& $Python (Join-Path $PSScriptRoot 'scripts\flow_step_ab.py') `
    --repo $Repo `
    --model $Model `
    --prompt-wav (Join-Path $Repo 'asset\zero_shot_prompt.wav') `
    --output $Output `
    --steps 10 4 2
if ($LASTEXITCODE -ne 0) {
    throw 'Stage 3 Flow step comparison failed'
}
