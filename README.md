# Fun-CosyVoice

**CosyVoice3 MNN 手机端本地语音合成应用**

基于阿里巴巴 [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 第三代模型的 Android arm64 移植，使用阿里 MNN 推理引擎，完全在手机本地运行，无需联网。

## 功能

- 文本转语音（TTS）：输入中文文本，手机本地合成 24kHz 单声道 WAV
- 零样本音色复刻：导入音色档案 ZIP，一键切换不同说话人
- 手机创建音色：从 MP3/WAV 截取 3-5 秒人声 → 提取语音特征 → 创建自定义音色
- 指令演绎（Instruct2）：通过自然语言控制语速、情感、方言（四川话、广东话、东北话等）
- GPU 加速：Flow 阶段使用 Adreno OpenCL，HiFT 使用 CPU 6 线程
- 模型管理：支持 ZIP 导入/导出，SHA-256 完整性校验

## 系统要求

- Android 8.0+（API 26+）
- arm64-v8a 处理器（骁龙 8 Gen 3 / Adreno 840 真机验证通过）
- **至少 6 GB 空闲存储空间**（模型 ~1.3 GB + 音色创建扩展 ~1.0 GB）
- 推荐 8 GB+ RAM

## 性能（BKQ_AN90 真机实测）

| 指标 | 热态 | 冷态 |
|------|------|------|
| 实时系数（RTF） | 0.79 - 0.96 | ~1.7 |
| LLM 耗时 | ~1.6 秒 | ~1.6 秒 |
| Flow GPU 耗时 | 0.8 - 1.1 秒 | ~3.4 秒 |
| HiFT CPU 耗时 | 1.4 - 2.0 秒 | ~1.9 秒 |
| 进程内存 | ~947 MB | ~947 MB |

## 快速开始

### 1. 编译安装

```bash
# 设置 JDK 17
export JAVA_HOME=/path/to/jdk-17

# 编译 Debug APK
./gradlew :app:assembleDebug

# 安装到手机
adb install app/build/outputs/apk/debug/app-debug.apk
```

### 2. 导入模型

打开应用后，需要导入两个必需的文件：

1. **MNN 模型包** (`cosyvoice3-mnn-mobile-fp16-complete.zip`, ~1.3 GB)
   - 点击"导入 ZIP" → 选择该文件
   - 17 个模型文件，SHA-256 自动校验
   - 导入后显示"已安装 · 1.30 GiB"

2. **音色创建扩展包** (`cosyvoice3-mnn-enrollment-extension.zip`, ~1.0 GB)
   - 点击"导入创建扩展" → 选择该文件
   - 包含语音 Tokenizer 和说话人特征提取模型
   - 导入后即可使用"手机创建音色"功能

> 两个 ZIP 文件可从 [GitHub Releases](https://github.com/fun/cosyvoice/releases) 下载。

### 3. 试听

1. 默认内置基准音色，可直接输入文字点击"合成并试听"
2. 可切换 GPU 调度模式（Flow: Buffer/Image, HiFT: CPU/GPU）
3. 可切换普通复刻 / 指令演绎模式

## 音色创建流程

1. 选择一段 MP3/WAV 音频文件
2. 设置截取时间（推荐 3-5 秒单人清晰语音）
3. 填写音色名称和截取片段对应文字
4. 点击"创建并选中音色"
5. 等待提取完成（约 30-60 秒）

## 项目结构

```
Fun-CosyVoice/
├── app/
│   ├── build.gradle.kts          # Android 应用配置
│   ├── src/main/
│   │   ├── AndroidManifest.xml
│   │   ├── java/com/fun/cosyvoice/
│   │   │   ├── MainActivity.kt              # Compose UI 主界面
│   │   │   ├── CosyVoiceRuntime.kt          # 合成管线编排（LLM→Flow→HiFT）
│   │   │   ├── CosyVoiceStore.kt            # 模型/音色档案管理
│   │   │   ├── CosyVoiceVoiceProfile.kt     # 音色档案数据模型
│   │   │   ├── CosyVoiceInstruction.kt      # Instruct2 指令预设
│   │   │   ├── CosyVoiceAudioDecoder.kt     # MediaCodec 音频解码
│   │   │   ├── CosyVoiceModelDownloader.kt  # HuggingFace 在线下载
│   │   │   ├── CosyVoiceLlmNative.kt        # LLM JNI 接口
│   │   │   ├── CosyVoiceFlowNative.kt       # Flow JNI 接口
│   │   │   ├── CosyVoiceHiFTNative.kt       # HiFT JNI 接口
│   │   │   └── CosyVoiceEnrollmentNative.kt # 音色注册 JNI 接口
│   │   └── jniLibs/arm64-v8a/
│   │       ├── libcosy_llm_jni.so           # LLM MNN 推理 (6.9 MB)
│   │       ├── libcosy_flow_jni.so          # Flow MNN 推理 (6.9 MB)
│   │       ├── libcosy_hift_jni.so          # HiFT MNN 推理 (7.0 MB)
│   │       ├── libcosy_enrollment_jni.so    # 音色注册 (7.0 MB)
│   │       └── libcosy_conditioner_exec.so  # Conditioner 预处理 (4.9 MB)
├── gradle/
│   └── libs.versions.toml         # 依赖版本目录
├── build.gradle.kts               # 根构建脚本
└── settings.gradle.kts            # 项目设置
```

## 技术栈

- **Kotlin 2.3.10** / **AGP 8.13.2** / **JDK 17**
- **Jetpack Compose** (Material 3) — UI
- **Kotlin Coroutines** — 异步管线
- **OkHttp 5** — 模型下载
- **阿里 MNN** — LLM/Flow/HiFT 推理引擎
- **OpenCL** — Flow GPU 加速

## 模型信息

| 模型 | 框架 | 大小 | 许可证 |
|------|------|------|--------|
| CosyVoice3-0.5B (MNN) | MNN | ~1.3 GB | Apache 2.0 |
| Speech Tokenizer v3 | MNN | ~930 MB | Apache 2.0 |
| CAM++ Speaker | MNN | ~27 MB | Apache 2.0 |

模型来源：[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)

## 许可证

本项目代码以 **Apache 2.0** 许可证发布。

原生库（`.so` 文件）从 CrispASR 和 MNN 构建，适用其各自的许可证。

## 致谢

- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — 阿里巴巴 CosyVoice 模型
- [alibaba/MNN](https://github.com/alibaba/MNN) — 阿里巴巴 MNN 推理引擎
- 本项目从 [gedoor/Legado](https://github.com/gedoor/Legado)（阅读 Archive）的 cosytest 分支独立出来
