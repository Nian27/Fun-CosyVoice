param(
    [string]$Python = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Environment = Join-Path $Root '.venv-export'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python runtime not found: $Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $Environment 'Scripts\python.exe'))) {
    & $Python -m venv $Environment
}
$EnvPython = Join-Path $Environment 'Scripts\python.exe'
& $EnvPython -m pip install --disable-pip-version-check --upgrade pip
& $EnvPython -m pip install --disable-pip-version-check `
    --index-url https://download.pytorch.org/whl/cpu `
    torch==2.3.1
& $EnvPython -m pip install --disable-pip-version-check `
    numpy==1.26.4 scipy==1.13.1 librosa==0.10.2.post1 soundfile==0.12.1 `
    onnx==1.19.1 onnxruntime==1.23.2
if ($LASTEXITCODE -ne 0) {
    throw 'Stage 2 export environment setup failed'
}
& $EnvPython -c "import torch, onnx, onnxruntime, scipy; print('torch', torch.__version__); print('onnx', onnx.__version__); print('onnxruntime', onnxruntime.__version__); print('scipy', scipy.__version__)"
