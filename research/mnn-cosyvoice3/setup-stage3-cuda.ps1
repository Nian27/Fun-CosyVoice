param(
    [string]$Python = 'D:\anaconda3\envs\pytorch\python.exe',
    [string]$DependencyPath = 'E:\AndroidStudioProjects\cosyvoice3-distill-lab\python-deps',
    [string]$Repo = 'E:\AndroidStudioProjects\CosyVoice-main'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "CUDA Python not found: $Python"
}
New-Item -ItemType Directory -Force -Path $DependencyPath | Out-Null
$env:PYTHONPATH = "$DependencyPath;$Repo;$Repo\third_party\Matcha-TTS"
$Validation = @'
import torch
import torchaudio
import transformers
import onnxruntime
import hyperpyyaml
import whisper
import cosyvoice.flow.flow_matching
assert torch.cuda.is_available()
print('python/torch ready', torch.__version__, torchaudio.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
'@

& $Python -c $Validation
if ($LASTEXITCODE -eq 0) {
    exit 0
}

# --no-deps is deliberate: the source Conda environment owns CUDA Torch.
$Packages = @(
    'HyperPyYAML==1.2.3', 'ruamel.yaml==0.18.17', 'ruamel.yaml.clib==0.2.15',
    'inflect==7.3.1', 'more-itertools==10.8.0', 'typeguard==4.5.2',
    'librosa==0.10.2', 'audioread==3.1.0', 'pooch==1.9.0', 'soxr==1.1.0',
    'msgpack==1.1.2', 'soundfile==0.12.1', 'modelscope==1.20.0',
    'omegaconf==2.3.0', 'antlr4-python3-runtime==4.9.3',
    'onnxruntime==1.18.0', 'coloredlogs==15.0.1', 'humanfriendly==10.0',
    'flatbuffers==25.12.19', 'protobuf==6.33.6',
    'openai-whisper==20231117', 'tiktoken==0.13.0',
    'transformers==4.51.3', 'huggingface-hub==0.36.2', 'regex==2026.1.15',
    'tokenizers==0.21.4', 'safetensors==0.7.0',
    'x-transformers==2.11.24', 'einops==0.8.2', 'einx==0.3.0', 'frozendict==2.4.7',
    'conformer==0.3.2', 'diffusers==0.29.0', 'hydra-core==1.3.2',
    'lightning==2.2.4', 'lightning-utilities==0.11.2',
    'torchmetrics==1.3.2', 'pytorch-lightning==2.2.4',
    'rich==13.7.1', 'gdown==5.1.0', 'wget==3.2', 'pyworld==0.3.4',
    'opencc-python-reimplemented==0.1.7'
)
& $Python -m pip install --disable-pip-version-check `
    --index-url http://mirrors.aliyun.com/pypi/simple/ `
    --trusted-host mirrors.aliyun.com --retries 10 --timeout 120 `
    --target $DependencyPath --upgrade --no-deps --no-build-isolation @Packages
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install isolated CosyVoice inference dependencies'
}

& $Python -c $Validation
if ($LASTEXITCODE -ne 0) {
    throw 'The CUDA environment or isolated dependencies failed validation'
}
