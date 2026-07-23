# MNN JNI 原生代码

这 4 个 JNI 包装文件 + 8 个 benchmark 工具是 CosyVoice3 在 Android 上运行的核心 C++ 层。

## 编译环境

- **MNN 3.6.0**（阿里 MNN 推理引擎）
- **Android NDK r26d**（交叉编译到 arm64-v8a）
- **CMake 3.31+** + Ninja

## 4 个 JNI 包装

| 文件 | 暴露的 JNI 函数 | 对应 Kotlin 文件 |
|------|----------------|-----------------|
| `CosyVoiceLlmJni.cpp` | `run()` / `reset()` | `CosyVoiceLlmNative.kt` |
| `CosyVoiceFlowJni.cpp` | `run()` / `reset()` | `CosyVoiceFlowNative.kt` |
| `CosyVoiceHiFTJni.cpp` | `run()` / `reset()` | `CosyVoiceHiFTNative.kt` |
| `CosyVoiceEnrollmentJni.cpp` | `enroll()` | `CosyVoiceEnrollmentNative.kt` |

## 8 个 Benchmark 工具

独立可执行文件，用于逐模块验证 MNN 推理的速度和数值正确性：

| 文件 | 功能 |
|------|------|
| `CosyVoiceLlmBenchmark.cpp` | LLM 单次推理基准 |
| `CosyVoiceLlmPersistentBenchmark.cpp` | LLM 常驻连续请求基准 |
| `CosyVoiceFlowBenchmark.cpp` | Flow 单次 estimator 基准 |
| `CosyVoiceFlowPersistentBenchmark.cpp` | Flow 常驻连续序列基准 |
| `CosyVoiceTwoStepFlowBenchmark.cpp` | 两步蒸馏 Flow 基准 |
| `CosyVoiceFlowConditionerBenchmark.cpp` | Conditioner 预处理基准 |
| `CosyVoiceHiFTBenchmark.cpp` | HiFT 合成基准 |
| `CosyVoiceHiFTE2EBenchmark.cpp` | HiFT 端到端（mel→WAV）基准 |
| `CosyVoiceHiFTPersistentBenchmark.cpp` | HiFT 常驻连续基准 |
| `CosyVoiceSpeechTokenizerBenchmark.cpp` | 语音 Tokenizer 注册基准 |

## 编译命令

```powershell
# 设置环境变量
$NdkRoot = 'C:\path\to\android-ndk-r26d'
$MnnSource = 'E:\path\to\MNN-master'
$BuildDir = '.\build\mnn-3.6-android-arm64-opencl'

# CMake 配置（交叉编译 arm64-v8a / OpenCL）
cmake -S $MnnSource -B $BuildDir -G Ninja `
    -DCMAKE_TOOLCHAIN_FILE="$NdkRoot\build\cmake\android.toolchain.cmake" `
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 `
    -DCMAKE_BUILD_TYPE=Release

# 编译所有 JNI 库
cmake --build $BuildDir `
    --target cosy_llm_jni cosy_flow_jni cosy_hift_jni cosy_enrollment_jni `
    --parallel 6
```

输出 .so 文件在 `$BuildDir/tools/cpp/` 下，复制到 `app/src/main/jniLibs/arm64-v8a/` 即可打包到 APK。

## 关键依赖

- **libMNN.so**：MNN 框架本体
- **libOpenCL.so**：高通 Adreno GPU 加速（非必须，但 Flow 阶段强烈建议用）
- **libcosy_conditioner_exec.so**：Conditioner 独立可执行文件（非 JNI，通过 ProcessBuilder 调用）

## 许可证

这些 JNI 包装文件基于 CrispASR v0.8.10 的 CosyVoice3 C API 编写，属于薄封装层。CrispASR 和 MNN 均为开源项目，适用各自的许可证。
