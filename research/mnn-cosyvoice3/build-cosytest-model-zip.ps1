param(
    [string]$Output = 'E:\AndroidStudioProjects\cosyvoice3-mnn-mobile-fp16-complete.zip'
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression

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

foreach ($File in $Files) {
    if (-not (Test-Path -LiteralPath $File.Source -PathType Leaf)) {
        throw "Model file not found: $($File.Source)"
    }
}

$outputDirectory = Split-Path -Parent $Output
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$temporaryOutput = "$Output.building"
Remove-Item -LiteralPath $temporaryOutput -Force -ErrorAction SilentlyContinue

$manifestFiles = foreach ($File in $Files) {
    $item = Get-Item -LiteralPath $File.Source
    [ordered]@{
        name = $File.Name
        bytes = $item.Length
        sha256 = (Get-FileHash -LiteralPath $File.Source -Algorithm SHA256).Hash
    }
}
$manifest = [ordered]@{
    format = 'cosyvoice3-mnn-mobile-model'
    formatVersion = 1
    sampleRate = 24000
    files = $manifestFiles
}

$stream = [System.IO.File]::Open($temporaryOutput, [System.IO.FileMode]::CreateNew)
try {
    $archive = [System.IO.Compression.ZipArchive]::new(
        $stream,
        [System.IO.Compression.ZipArchiveMode]::Create,
        $false
    )
    try {
        $index = 0
        foreach ($File in $Files) {
            $index++
            Write-Host "[$index/$($Files.Count)] $($File.Name)"
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive,
                $File.Source,
                $File.Name,
                [System.IO.Compression.CompressionLevel]::NoCompression
            ) | Out-Null
        }
        $entry = $archive.CreateEntry('manifest.json', [System.IO.Compression.CompressionLevel]::NoCompression)
        $writer = [System.IO.StreamWriter]::new($entry.Open(), [System.Text.UTF8Encoding]::new($false))
        try {
            $writer.Write(($manifest | ConvertTo-Json -Depth 5))
        } finally {
            $writer.Dispose()
        }
    } finally {
        $archive.Dispose()
    }
} finally {
    $stream.Dispose()
}

Move-Item -LiteralPath $temporaryOutput -Destination $Output -Force

$readStream = [System.IO.File]::OpenRead($Output)
try {
    $archive = [System.IO.Compression.ZipArchive]::new(
        $readStream,
        [System.IO.Compression.ZipArchiveMode]::Read,
        $false
    )
    try {
        foreach ($File in $Files) {
            $entry = $archive.GetEntry($File.Name)
            if ($null -eq $entry) { throw "ZIP entry missing: $($File.Name)" }
            $expectedLength = (Get-Item -LiteralPath $File.Source).Length
            if ($entry.Length -ne $expectedLength) {
                throw "ZIP entry size mismatch: $($File.Name) $($entry.Length) / $expectedLength"
            }
            $expectedHash = ($manifestFiles | Where-Object { $_.name -eq $File.Name }).sha256
            $hashAlgorithm = [System.Security.Cryptography.SHA256]::Create()
            $entryStream = $entry.Open()
            try {
                $actualHash = [Convert]::ToHexString($hashAlgorithm.ComputeHash($entryStream))
            } finally {
                $entryStream.Dispose()
                $hashAlgorithm.Dispose()
            }
            if ($actualHash -ne $expectedHash) {
                throw "ZIP entry hash mismatch: $($File.Name)"
            }
        }
        if ($archive.Entries.Count -ne ($Files.Count + 1)) {
            throw "Unexpected ZIP entry count: $($archive.Entries.Count)"
        }
    } finally {
        $archive.Dispose()
    }
} finally {
    $readStream.Dispose()
}

$item = Get-Item -LiteralPath $Output
$hash = (Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash
Write-Host "Created: $($item.FullName)"
Write-Host "Bytes: $($item.Length)"
Write-Host "SHA256: $hash"
