param(
    [string]$SourceDirectory = "$PSScriptRoot\models\voice-enrollment",
    [string]$OutputZip = "E:\AndroidStudioProjects\cosyvoice3-mnn-enrollment-extension.zip"
)

$ErrorActionPreference = "Stop"

$files = @(
    @{ Name = "speech-tokenizer-v3.fp32.inline.mnn"; Bytes = 969762652; Sha256 = "EB436E54A7A4227059F7A61C47DA57EFE2BF1124C89CD24BD2841BBC190A4769" },
    @{ Name = "campplus.fp32.mnn"; Bytes = 607208; Sha256 = "7F9889B437DABB6A906CF6BC9244C2AA9A0AC6599C602D8ECE31183EA7A5D73D" },
    @{ Name = "campplus.fp32.mnn.weight"; Bytes = 27375488; Sha256 = "51D633D3B49AA532080DF1015AC8DF87DA819187C1D9049DE39DAB54E1A4A40B" },
    @{ Name = "flow-speaker-affine-weight.bin"; Bytes = 61440; Sha256 = "0131B531DDBB1A6120DEE9C0832EDE4D1F76EB73B776F13EC890C522E75350B6" },
    @{ Name = "flow-speaker-affine-bias.bin"; Bytes = 320; Sha256 = "B47EFEAA84822ABE17D7E6308AB9AF189F29A0CB86216D564D9BF4B137B24849" }
)

foreach ($spec in $files) {
    $path = Join-Path $SourceDirectory $spec.Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing enrollment model: $path"
    }
    $item = Get-Item -LiteralPath $path
    if ($item.Length -ne $spec.Bytes) {
        throw "Size mismatch for $($spec.Name): $($item.Length) / $($spec.Bytes)"
    }
    $actualHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($actualHash -ne $spec.Sha256) {
        throw "SHA-256 mismatch for $($spec.Name): $actualHash"
    }
}

$outputDirectory = Split-Path -Parent $OutputZip
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$temporary = "$OutputZip.building"
Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$stream = [System.IO.File]::Open($temporary, [System.IO.FileMode]::CreateNew)
try {
    $archive = [System.IO.Compression.ZipArchive]::new(
        $stream,
        [System.IO.Compression.ZipArchiveMode]::Create,
        $false
    )
    try {
        foreach ($spec in $files) {
            $source = Join-Path $SourceDirectory $spec.Name
            $entry = $archive.CreateEntry($spec.Name, [System.IO.Compression.CompressionLevel]::NoCompression)
            $entryStream = $entry.Open()
            try {
                $sourceStream = [System.IO.File]::OpenRead($source)
                try {
                    $sourceStream.CopyTo($entryStream, 1048576)
                } finally {
                    $sourceStream.Dispose()
                }
            } finally {
                $entryStream.Dispose()
            }
        }
    } finally {
        $archive.Dispose()
    }
} finally {
    $stream.Dispose()
}

Move-Item -LiteralPath $temporary -Destination $OutputZip -Force
Write-Output $OutputZip
Write-Output "SHA256=$((Get-FileHash -LiteralPath $OutputZip -Algorithm SHA256).Hash)"
Write-Output "Bytes=$((Get-Item -LiteralPath $OutputZip).Length)"
