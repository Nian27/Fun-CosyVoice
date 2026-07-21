param(
    [string]$MnnSource = 'E:\AndroidStudioProjects\MNN-master',
    [string]$NdkRoot = 'C:\Users\Administrator\Documents\阅读archive\research\android-ndk-r26d'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $Root 'build\mnn-3.6-android-arm64-opencl'
$CMake = 'C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$Ninja = 'C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
$Toolchain = Join-Path $NdkRoot 'build\cmake\android.toolchain.cmake'

foreach ($Required in @($CMake, $Ninja, $Toolchain)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required build tool not found: $Required"
    }
}

& $CMake -S $MnnSource -B $BuildDir -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$Ninja" `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    '-DCMAKE_BUILD_TYPE=Release' `
    '-DANDROID_ABI=arm64-v8a' `
    '-DANDROID_PLATFORM=android-26' `
    '-DANDROID_STL=c++_static' `
    '-DMNN_BUILD_SHARED_LIBS=OFF' `
    '-DMNN_BUILD_TOOLS=ON' `
    '-DMNN_BUILD_CONVERTER=OFF' `
    '-DMNN_BUILD_TEST=OFF' `
    '-DMNN_BUILD_BENCHMARK=OFF' `
    '-DMNN_BUILD_DEMO=OFF' `
    '-DMNN_OPENCL=ON' `
    '-DMNN_VULKAN=OFF' `
    '-DMNN_ARM82=ON' `
    '-DMNN_LOW_MEMORY=ON' `
    '-DMNN_SEP_BUILD=OFF' `
    '-DMNN_USE_LOGCAT=OFF' `
    '-DMNN_SUPPORT_TRANSFORMER_FUSE=ON'
if ($LASTEXITCODE -ne 0) {
    throw "MNN Android configure failed with exit code $LASTEXITCODE"
}

& $CMake --build $BuildDir --target CosyVoiceFlowBenchmark.out --parallel 6
if ($LASTEXITCODE -ne 0) {
    throw "MNN Android build failed with exit code $LASTEXITCODE"
}

$Executable = Join-Path $BuildDir 'CosyVoiceFlowBenchmark.out'
if (-not (Test-Path -LiteralPath $Executable)) {
    throw "Android benchmark was not produced: $Executable"
}
Write-Host "Android benchmark: $Executable"
