# Fun-CosyVoice

**CosyVoice3 MNN 手机端本地语音合成 — 纯 Android arm64 独立应用**

从 [Legado / 阅读 Archive](https://github.com/gedoor/Legado) 的 `cosytest` 分支独立出来的 CosyVoice3 本地 TTS App。使用阿里 MNN 推理引擎，全部在手机本地运行，无需联网。

---

## 目录

- [项目背景：为什么要做这个](#项目背景为什么要做这个)
- [整体架构](#整体架构)
- [合成管线详解（4 个阶段）](#合成管线详解4-个阶段)
- [问题记录（供后续开发者参考）](#问题记录供后续开发者参考)
- [性能数据（BKQ_AN90 真机实测）](#性能数据bkq_an90-真机实测)
- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [模型管理](#模型管理)
- [音色管理](#音色管理)
- [手机创建音色](#手机创建音色)
- [GPU / CPU 后端切换](#gpu--cpu-后端切换)
- [项目结构](#项目结构)
- [从源码构建](#从源码构建)
- [常见问题 / Troubleshooting](#常见问题--troubleshooting)
- [技术栈](#技术栈)
- [模型信息](#模型信息)
- [欢迎参与](#欢迎参与)
- [许可证](#许可证)

---

## 项目背景：为什么要做这个

**起点**：Legado / 阅读 Archive 的 `cosytest` 分支内置了 CosyVoice3 本地 TTS 合成。但存在几个严重问题：

1. **只在高通（Qualcomm Adreno）手机上能跑** — `libOpenCL.so required="true"` 强制 OpenCL，华为麒麟 / 联发科天玑 / 三星 Exynos 全部闪退
2. **集成在阅读 App 内部** — 不方便单独测试和调试，且与阅读的主业务流程耦合
3. **Flow 阶段硬编码 OpenCL GPU** — 没有 CPU fallback，非高通设备无法合成

**这个独立 App 的目标**：

- ✅ 去掉所有阅读 App 的业务代码，只保留 CosyVoice3 合成和音色管理
- ✅ 添加 CPU / OpenCL 双后端自动检测
- ✅ 提供完整的 ZIP 导入/导出模型系统和音色档案管理
- ✅ 保留手机创建音色（Enrollment）功能
- ✅ 开源，欢迎大家一起来修
- ⚠️ **注意**：非高通设备能否用 CPU 合成还需要实测验证

---

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                     MainActivity (Compose UI)             │
│  模型管理 │ 音色选择 │ GPU调度 │ 试听 │ 创建音色 │ 下载    │
└───────────────┬─────────────────────────┬───────────────┘
                │                         │
                ▼                         ▼
       ┌────────────────┐      ┌──────────────────────┐
       │  CosyVoiceStore │      │  CosyVoiceRuntime     │
       │  模型/音色/注册   │      │  合成管线编排 (Mutex)  │
       │  ZIP导入/导出   │      │  LLM→Cond→Flow→HiFT  │
       └────────────────┘      └──────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────┐
          ▼                             ▼                     ▼
┌───────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│ CosyVoiceLlmNative│   │ CosyVoiceFlowNative  │   │ CosyVoiceHiFTNative │
│ (libcosy_llm_jni) │   │ (libcosy_flow_jni)   │   │ (libcosy_hift_jni)  │
├───────────────────┤   ├─────────────────────┤   ├─────────────────────┤
│ MNN LLM           │   │ MNN Flow            │   │ MNN HiFT            │
│ CPU 推理          │   │ CPU/OpenCL 推理      │   │ CPU/OpenCL 推理     │
│ 生成语音 Token     │   │ Token → Mel 频谱     │   │ Mel → WAV 波形      │
└───────────────────┘   └─────────────────────┘   └─────────────────────┘
                                                 ┌──────────────────────┐
                                                 │CosyVoiceEnrollmentNtv│
                                                 │(libcosy_enrollment)  │
                                                 │ 音色注册（可选扩展） │
                                                 └──────────────────────┘
```

**5 个原生 .so 文件**（`app/src/main/jniLibs/arm64-v8a/`，共 32.7 MB）：

| 文件 | 大小 | 功能 |
|------|------|------|
| `libcosy_llm_jni.so` | 6.9 MB | LLM 推理：从文字生成语音 Token（CPU 4 线程） |
| `libcosy_flow_jni.so` | 6.9 MB | Flow 模型：Token → Mel 频谱（支持 CPU/OpenCL） |
| `libcosy_hift_jni.so` | 7.0 MB | HiFT 模型：Mel → WAV 波形（支持 CPU/OpenCL） |
| `libcosy_enrollment_jni.so` | 7.0 MB | 音色注册（可选扩展，约 974 MB 模型） |
| `libcosy_conditioner_exec.so` | 4.9 MB | Conditioner 预处理（独立子进程） |

---

## 合成管线详解（4 个阶段）

每次合成一句文字要走 4 个阶段，顺序执行。故障排查时按这个顺序看日志：

### 阶段 1：LLM — 文字 → 语音 Token

```
输入: "你好，欢迎使用阅读"
        │
        ▼
    [prompt-speech-tokens.csv]  ← 当前选中音色的参考 Token（零样本复刻才有）
        │
        ▼
    LLM (libcosy_llm_jni / CPU 4线程 / llm.mnn + llm.mnn.weight ~336 MB)
        │
        ▼
输出: speech-tokens-0.csv  ← 一串数字，每个 0~6560 表示一个语音码本索引
```

- 耗时：约 1.5~2.0 秒
- 如果是"指令演绎"（Instruct2）模式，不附加参考 Token，用指令文字替代
- LLM 是标准的 Transformer 结构，MNN 跑在 CPU 上

### 阶段 2：Conditioner — Token 预处理

```
输入: speech-tokens-0.csv + 音色档案（prompt-cond.bin / spks.bin）
        │
        ▼
    Conditioner (libcosy_conditioner_exec.so / 独立子进程 / flow-conditioner.fp32.mnn)
        │
        ▼
输出: flow-input/ 目录下的 Mel 频谱碎片文件
```

- 独立 C++ 可执行文件，通过 ProcessBuilder 启动
- 负责将 Token 和音色信息合并成 Flow 模型的输入格式
- 如果 Conditioner 失败，日志在 run-{timestamp}/conditioner.log

### 阶段 3：Flow — Token → Mel 频谱

```
输入: flow-input/ → Conditioner 输出的特征
        │
        ▼
    Flow (libcosy_flow_jni / CPU 6线程 或 OpenCL GPU / flow.fp16.mnn ~633 MB)
        │
        ▼
输出: student_target_mel_android.bin  ← Mel 频谱数据
```

- **这是最复杂的阶段**：MNN FP16 模型，~633 MB 权重
- **Flow 后端可选**：CPU（`flowBackend="cpu"`）或 OpenCL GPU（`flowBackend="opencl"`）
- CPU 模式：`precision="normal"`, `threads=6`
- OpenCL 模式：`precision="high"`, mode 可选 4(自动)/68(Buffer)/132(Image)
- Flow 输出是 80 通道的 Mel 频谱，帧率 50fps（每帧 20ms）
- GPU 编译缓存存在 `gpu-cache/flow-fp16-{backend}-{precision}-mode{mode}.cache`

### 阶段 4：HiFT — Mel → WAV 波形

```
输入: mel.bin（上一步的 Mel 频谱）+ source-linear-weight.bias
        │
        ▼
    HiFT (libcosy_hift_jni / CPU 6线程 或 OpenCL GPU / hift-core.fp32.mnn ~67 MB + hift-f0.fp32.mnn ~12.6 MB)
        │
        ▼
输出: hift-android.wav ← 24kHz 单声道 16-bit PCM WAV
```

- HiFT 其实就是 HiFi-GAN + F0 预测
- 支持 CPU 和 OpenCL 后端
- 输出文件 > 44 字节才算成功

---

## 问题记录（供后续开发者参考）

> ⚠️ 以下 fix **只在我的手机（骁龙 8 Gen 3）上测试过**。非高通手机能否正常工作**还不确定**，欢迎有麒麟/天玑设备的朋友帮忙验证并提交反馈。

### 1. 非高通手机 OpenCL 闪退（不确定是否完全解决）

**症状**：华为麒麟 / 联发科天玑手机安装后闪退，Logcat 看到 `libOpenCL.so not found`

**根因**：AndroidManifest.xml 中 `<uses-native-library android:name="libOpenCL.so" android:required="true"/>` — 只有 Qualcomm Adreno GPU 有这个库

**尝试的修复（2 处改动，未在非高通设备上实测）**：
- 原本 cosytest 的 Manifest 有 `required="true"`，独立 App 新建 Manifest 时直接没加这一行
- 运行时检测：`CosyVoiceRuntime.detectBestFlowBackend()` 用 `System.loadLibrary("OpenCL")` 检测，没有就降级到 CPU
- **但我无法确认 Flow CPU 的 libcosy_flow_jni.so 是否真的能在非高通设备上运行**

关联代码：`CosyVoiceRuntime.kt:71-82` → `detectBestFlowBackend()`

### 2. Flow 阶段只支持 OpenCL，没有 CPU fallback

**症状**：非高通手机上 Flow 阶段直接崩溃

**根因**：MNN Flow JNI 的 `backend` 参数硬编码为 `"opencl"`

**修复**：`CosyVoiceSynthesisOptions` 新增 `flowBackend` 字段，`CosyVoiceFlowNative.run()` 把 backend 参数传下去。CPU 模式下自动设置 `precision="normal"`, `threads=6`

关联代码：`CosyVoiceRuntime.kt:14` → `flowBackend` 默认值

### 3. `extractNativeLibs` 配置问题导致 .so 加载失败

**症状**：`System.loadLibrary("cosy_llm_jni")` 抛 `UnsatisfiedLinkError`

**根因**：原本 cosytest 的 Android 12+ 设备上 `extractNativeLibs=false`，.so 未被提取到 native lib 目录

**修复**：在 cosytest 的 `app/build.gradle.kts` 中添加了 `jniLibs.useLegacyPackaging = true`。独立 App 也已加上此配置。

### 4. 首次打开 App 空指针崩溃

**症状**：打开 App 后未导入模型时，点击"试听"崩溃

**根因**：`refresh()` 调用 `selectVoiceProfile()`，而 `loadVoiceProfile()` 内部 `check(modelStatus().ready)` 在模型未就绪时抛异常

**修复**：`CosyVoiceStore.selectVoiceProfile()` 加 `if (modelStatus().ready)` 保护

关联代码：`CosyVoiceStore.kt:87-92`

### 5. 导出模型时 ContentProvider URI 残留

**症状**：模型导出失败后，uri 被 ContentResolver 删除，但实际上文件没写完整（或者相反）

**根因**：异常处理中 `delete()` 调用不对

**修复**：用 try-catch 包裹写入逻辑，失败时显式删除不完整的 uri

### 6. GPU 编译缓存没有清理干净

**症状**：切换 Flow backend 后首次合成特别慢（30+ 秒），之后正常

**根因**：不同 backend/mode 的 GPU 编译缓存是分开的，但旧缓存不会被自动清理

**当前状态**：删除模型时会一并删除 `gpu-cache/` 目录。手动切换 backend 后可以手动删一次。

---

## 性能数据（BKQ_AN90 真机实测）

测试设备：BKQ_AN90（骁龙 8 Gen 3 / Adreno 840 / 12 GB RAM）

| 指标 | 热态（连续合成） | 冷态（首次合成） |
|------|------------------|------------------|
| **实时系数（RTF）** | **0.79 - 0.96** | **~1.7** |
| LLM 耗时 | ~1.6 秒 | ~1.6 秒 |
| Flow GPU 耗时 | 0.8 - 1.1 秒 | ~3.4 秒（GPU 编译） |
| Flow CPU 耗时 | ~2.5 - 3.5 秒（预估） | ~3.5 - 4.0 秒（预估） |
| HiFT CPU 耗时 | 1.4 - 2.0 秒 | ~1.9 秒 |
| 进程内存 | ~947 MB | ~947 MB |

> **非高通手机**：Flow CPU 模式的性能预期比上面 GPU 数据慢 2-3 倍，实测 RTF 可能在 2~3 之间。整体 10 秒文本的合成时间可能在 8~15 秒，仍然可用但不如高通流畅。

---

## 系统要求

| 项目 | 要求 |
|------|------|
| Android | API 26+（Android 8.0+） |
| 架构 | arm64-v8a 处理器 |
| RAM | 推荐 8 GB+（合成时占用 ~1 GB） |
| **存储（模型）** | **至少 1.4 GB**（模型包 ~1.3 GB + 运行时临时文件） |
| **存储（创建音色）** | **额外 ~974 MB**（可选扩展，不安装不影响合成） |
| GPU（可选） | 任何 OpenCL 支持的 GPU（Adreno / Mali / PowerVR） |

---

## 快速开始

### 1. 下载 APK

从 [GitHub Releases](https://github.com/Nian27/Fun-CosyVoice/releases) 下载最新的 `app-debug.apk`

```bash
# 或者自己构建（见"从源码构建"章节）
```

### 2. 安装 APK

```bash
adb install -r app-debug.apk
```

### 3. 下载模型

两个必需的文件（从 GitHub Releases 下载）：

| 文件 | 大小 | 说明 |
|------|------|------|
| `cosyvoice3-mnn-mobile-fp16-complete.zip` | ~1.3 GB | 完整 MNN 模型包（17 个文件） |
| `cosyvoice3-mnn-enrollment-extension.zip` | ~974 MB | 音色创建扩展（可选，如需手机创建音色） |

### 4. 导入模型

打开 App：

1. **导入 MNN 模型包** → 点击"导入 ZIP" → 选择 `cosyvoice3-mnn-mobile-fp16-complete.zip`
   - 自动校验 17 个文件的 SHA-256
   - 导入完成后状态显示"已安装 · 1.30 GiB"
2. **（可选）导入音色创建扩展** → 点击"导入创建扩展" → 选择 `cosyvoice3-mnn-enrollment-extension.zip`
   - 导入后即可使用"手机创建音色"功能

### 5. 试听

1. 默认已选中"基准音色"，直接输入文字
2. 点击"合成并试听"
3. 等待 3-10 秒即可听到合成的语音

---

## 模型管理

### 文件清单

模型包包含 17 个文件，共 ~1.3 GB。存储在 `app内部存储/files/cosyvoice3-mnn/model/` 下：

| 文件 | 大小 | 作用 |
|------|------|------|
| `llm.mnn.weight` | 336.5 MB | LLM 权重（最大的单个文件） |
| `embeddings_bf16.bin` | 271.2 MB | LLM 嵌入层 |
| `flow.cfg-student-2step.batch1.fp16.mnn.weight` | 632.4 MB | Flow 模型权重 |
| `rand-noise.bin` | 4.58 MB | Flow 共享噪声 |
| `flow-conditioner.fp32.mnn` | 4.20 MB | Conditioner 模型 |
| `prompt-speech-tokens.csv` | ~400 B | 基准音色 Token |
| `prompt-cond.bin` | ~54 KB | 基准音色 Condition |
| `spks.bin` | 320 B | 基准音色 Speaker 向量 |
| `source-linear-weight.bias` | 36 B + 4 B | 线性映射参数 |
| 其余文件 | 几 KB ~ 几十 MB | 配置文件、Tokenizer、F0 模型等 |

### ZIP 导入机制

1. 读取 ZIP 中的每个文件，文件名匹配 `MODEL_FILE_SPECS` 列表
2. 解压到 staging 目录
3. 校验文件大小和 SHA-256
4. 逐个移动到 model 目录（原子操作：先写 `.importing` 再 rename）

### 导出

选择"导出 ZIP" → 选择保存位置 → 生成无压缩的 ZIP 包（~1.3 GB）

---

## 音色管理

### 音色档案结构

每个音色是一个目录，存储在 `.../voices/{profile-id}/`：

```
profile.json              # 元数据：名称、Token数、Hash等
prompt-speech-tokens.csv  # 参考语音 Token（逗号分隔的数字）
prompt-cond.bin           # 参考语音 Conditioner 输出（80ch × frame × 4B）
spks.bin                  # 说话人嵌入向量（80ch × 4B = 320B）
source.wav（可选）        # 创建音色时的原始参考音频
rand-noise.bin（symlink） # 符号链接到模型目录的共享噪声
```

### 内置基准音色

- ID: `builtin-mnn-reference-v1`
- 87 Token，174 帧
- 从模型包中的 `prompt-speech-tokens.csv` + `prompt-cond.bin` + `spks.bin` 构建
- 不可删除

### 导入音色 ZIP

支持批量导入（选多个 ZIP 或一个包含多个音色的 ZIP）：

1. 点击"批量导入 ZIP"
2. 选择音色档案 ZIP（单个或多个）
3. 自动校验兼容性（modelId 必须匹配）
4. 导入后自动选中最后一个导入的音色

### 导出音色

- 单个导出：点击对应音色的"导出"按钮
- 全部导出：点击"导出全部"（仅导出非内置音色）

### 实时音色

如果某个音色的 Token 数超过 125（比如从 MP3 创建的音色），合成时会报错。
点击"生成实时版"会：
1. 截取前 125 个 Token
2. 截取对应的 Conditioner 数据（80ch × 125 帧）
3. 生成一个新的音色档案（自动命名为"原名称（实时）"）

---

## 手机创建音色

需要先导入"音色创建扩展"（~974 MB）。

### 流程

1. **选择参考音频** → 点击"选择 MP3/音频"，选一段包含单人清晰语音的文件
2. **截取片段** → 填入起止秒数（推荐 3-5 秒，必须 3-15 秒）
3. **填写信息** → 音色名称 + 片段对应的文字（必须完全一致）
4. **创建** → 点击"创建并选中音色"
5. **等待** → 约 30-60 秒，后台运行：
   - 解码音频 → MediaCodec 硬解
   - 语音 Tokenizer（speech-tokenizer-v3.fp32.inline.mnn, ~924 MB）→ 提取 Token
   - CAM++ 说话人识别（campplus.fp32.mnn）→ 提取 Speaker 向量
   - 说话人特征投影 → 生成 speaker.bin
6. **完成** → 自动选中新音色，可直接合成试听

### 常见错误码

| 错误码 | 含义 |
|--------|------|
| 10 | 参考 WAV 文件无法读取 |
| 11 | 人声长度不在 3-15 秒 |
| 20-28 | 语音 Tokenizer 模型问题（检查扩展文件是否完整） |
| 30-37 | 说话人模型问题 |
| 40 | Token 超过 125 个限制 |
| 42 | 音色文件写入失败（检查存储空间） |

---

## GPU / CPU 后端切换

### Flow 后端

UI 上的"Flow 后端"可选择：

| 选项 | 说明 |
|------|------|
| **OpenCL（默认）** | GPU 加速，精度 `high`，推荐高通设备。自动检测可用性 |
| **CPU** | MNN CPU 后端，精度 `normal`，6 线程。适用于无 OpenCL 设备 |

### Flow OpenCL mode

当后端为 OpenCL 时，可选择 GPU 计算模式：

| 模式 | 值 | 说明 |
|------|-----|------|
| 自动 | 4 | 让 MNN 自行选择 |
| Buffer | 68 | 用 OpenCL Buffer 内存（兼容性最好） |
| Image | 132 | 用 OpenCL Image 内存（性能可能更好） |

### HiFT 后端

| 选项 | 说明 |
|------|------|
| **CPU（默认）** | 6 线程，稳妥 |
| OpenCL | GPU 加速，兼容性可能不如 Flow |

### 自动检测逻辑

```kotlin
fun detectBestFlowBackend(): String {
    return if (openClAvailable) "opencl" else "cpu"
}
```

检测方式：`System.loadLibrary("OpenCL")` — 如果能加载就认为是 OpenCL 可用。

---

## 项目结构

```
Fun-CosyVoice/
├── app/
│   ├── build.gradle.kts              # Android 应用配置
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/com/fun/cosyvoice/
│       │   ├── MainActivity.kt                  # Compose UI 主界面（552 行）
│       │   ├── CosyVoiceRuntime.kt              # 合成管线编排（291 行）
│       │   ├── CosyVoiceStore.kt                # 模型/音色/注册管理（619 行）
│       │   ├── CosyVoiceVoiceProfile.kt         # 音色档案数据模型（66 行）
│       │   ├── CosyVoiceInstruction.kt          # Instruct2 指令预设（38 行）
│       │   ├── CosyVoiceAudioDecoder.kt         # MediaCodec 音频解码（215 行）
│       │   ├── CosyVoiceModelDownloader.kt      # HuggingFace 在线下载（160 行）
│       │   ├── CosyVoiceLlmNative.kt            # LLM JNI 接口（18 行）
│       │   ├── CosyVoiceFlowNative.kt           # Flow JNI 接口（19 行）
│       │   ├── CosyVoiceHiFTNative.kt           # HiFT JNI 接口（21 行）
│       │   └── CosyVoiceEnrollmentNative.kt     # 音色注册 JNI 接口（42 行）
│       └── jniLibs/arm64-v8a/                   # 5 个 .so，共 32.7 MB
├── gradle/
│   └── libs.versions.toml           # 依赖版本目录
├── build.gradle.kts                  # 根构建脚本
├── settings.gradle.kts               # 项目设置
├── gradlew / gradlew.bat             # Gradle Wrapper
├── research/
│   └── mnn-cosyvoice3/              # 完整模型移植研究脚本（可复现的蒸馏/构建/基准）
│       ├── scripts/*.py              # Python：蒸馏训练、模型导出、数值验证
│       ├── *.ps1                     # PowerShell：构建、运行、基准测试
│       ├── STAGE3_FEASIBILITY.md     # Flow 蒸馏可行性报告
│       └── VOICE_ENROLLMENT*.md      # 音色创建方案
└── docs/
    └── DEVELOPMENT_STORY.md          # 开发全流程记录
```

### 核心代码行数统计

| 文件 | 行数 | 职责 |
|------|------|------|
| CosyVoiceStore.kt | 619 | 最核心：模型校验、ZIP 导入导出、音色管理、注册管理 |
| MainActivity.kt | 552 | Compose UI + Activity 生命周期 + 事件处理 |
| CosyVoiceRuntime.kt | 291 | 合成管线编排、Mutex 同步、性能报告 |
| CosyVoiceAudioDecoder.kt | 215 | MediaCodec 音频解码 + WAV 写入 |
| CosyVoiceModelDownloader.kt | 160 | HuggingFace 在线断点续传下载 |
| **合计** | **2,041** | |

---

## 从源码构建

### 环境要求

- JDK 17+
- Android SDK 36（compileSdk）
- Android NDK（仅编译原生库需要，已提供预编译 .so）

### 构建命令

```bash
# 1. 克隆
git clone https://github.com/Nian27/Fun-CosyVoice.git
cd Fun-CosyVoice

# 2. 构建 Debug APK
./gradlew :app:assembleDebug

# 3. 构建 Release APK（需配置签名）
./gradlew :app:assembleRelease

# 4. 安装
adb install app/build/outputs/apk/debug/app-debug.apk
```

### 编译原生库

预编译的 .so 已包含在 `app/src/main/jniLibs/arm64-v8a/`。如需自己编译：

1. 从 https://github.com/crisp-oss/mnn-cosyvoice-jni 获取源码
2. 配置 Android NDK
3. 用 CMake 交叉编译到 arm64-v8a
4. 替换 `jniLibs/arm64-v8a/` 下的对应文件

---

## 常见问题 / Troubleshooting

### Q：App 打开后闪退

1. 检查 Logcat：`adb logcat -s "CosyVoice*" "*:E"`
2. 确认 Android 8.0+ / arm64
3. 如果看到 `UnsatisfiedLinkError` → 确认 APK 包含 .so 文件（解压 APK 查看 `lib/arm64-v8a/`）

### Q：合成后没有声音 / 显示错误

1. **"LLM JNI 执行失败"** → 检查模型是否完整（"MNN 模型"卡片应该有绿色的"已安装"）
2. **"Flow JNI 执行失败"** → 检查日志 `run-{timestamp}/flow-report.jsonl`，常见原因：
   - OpenCL 设备不支持 → 手动切换 Flow 后端到 CPU
   - GPU 编译缓存损坏 → 删除模型重新导入
3. **"HiFT JNI 执行失败"** → 类似 Flow 的排查方法

### Q：合成很慢

- **冷态首次合成**：~10 秒正常，因为 GPU 需要编译 kernel
- **持续合成变慢**：检查是否有后台进程占用 CPU/GPU
- **非高通手机 **：Flow CPU 模式预计比 GPU 慢 2-3 倍，整体合成时间 ~8-15 秒

### Q："音色提示过长"错误

当前音色的 Token 数超过 125。解决方法：
- 点击该音色的"生成实时版"按钮
- 或用更短的参考音频重新创建音色

### Q：音色创建失败

1. 确认已导入音色创建扩展（~974 MB）
2. 确保截取 3-5 秒清晰单人语音
3. 确保填写的文字与音频内容一致
4. 查看 Logcat 中的错误码（见上面"常见错误码"表格）

### Q：导入 ZIP 时 SHA-256 校验失败

- ZIP 文件可能损坏 → 重新下载
- 文件被修改过 → 重新下载原始模型包
- 存储空间不足 → 确保有至少 1.4 GB 空闲

### Q：我想在非高通手机上测试

1. 安装 APK
2. 导入模型
3. 在"GPU 调度"卡片中把 Flow 后端手动切换到 CPU
4. 试听
5. **如果 CPU 模式也崩溃** → 请在 GitHub Issues 中提交 Logcat

---

## 技术栈

| 技术 | 用途 |
|------|------|
| **Kotlin 2.3.10** | 主语言 |
| **AGP 8.13.2** | Android Gradle 插件 |
| **JDK 17** | 编译 |
| **Jetpack Compose + Material 3** | UI 框架 |
| **Kotlin Coroutines** | 异步管线 / Mutex 同步 |
| **OkHttp 5** | HuggingFace 模型下载 |
| **阿里 MNN** | LLM / Flow / HiFT 推理引擎 |
| **OpenCL** | Flow / HiFT GPU 加速 |
| **MediaCodec** | 参考音频解码 |
| **Gradle Version Catalog** | 依赖管理 |

---

## 模型信息

| 模型 | 框架 | 大小 | 许可证 | 来源 |
|------|------|------|--------|------|
| CosyVoice3-0.5B (MNN) | MNN | ~1.3 GB | Apache 2.0 | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |
| Speech Tokenizer v3 | MNN | ~924 MB | Apache 2.0 | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |
| CAM++ Speaker | MNN | ~27 MB | Apache 2.0 | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |

模型下载链接：**[GitHub Releases 页面](https://github.com/Nian27/Fun-CosyVoice/releases)**

---

## 欢迎参与

**目前这个项目只在我的一台骁龙 8 Gen 3 手机上测试通过**。非高通设备（麒麟/天玑/Exynos）能不能跑、跑得怎么样，都需要大家帮忙验证。

### 你可以做什么

| 角色 | 能做 |
|------|------|
| **有非高通手机的人** | 安装 APK 试试能不能用，[提 Issue](https://github.com/Nian27/Fun-CosyVoice/issues/new) 告诉我结果 |
| **Android 开发者** | 修 Bug、优化性能、加功能，直接提 Pull Request |
| **对 TTS 感兴趣的人** | 读代码、提建议、分享你创建的音色档案 |
| **任何用户** | 用用看，不好用就骂，骂完记得告诉我哪里不好 |

### 我知道的问题（还没修）

1. **非高通 Flow CPU 兼容性**：我没麒麟/天玑设备，`libcosy_flow_jni.so` 的 CPU 后端能不能用**完全没有验证**
2. **首次加载慢**：冷态首次合成 GPU 需要编译 kernel，等待 10-15 秒是正常的
3. **没有国际化**：UI 只有中文
4. **APK 没有签名**：debug APK 直接安装，部分系统会提示"未知来源"

### 贡献方式

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/my-fix`）
3. 提交修改（`git commit -am 'fix: xxx'`）
4. 推送到分支（`git push origin feature/my-fix`）
5. 创建 Pull Request

---

## 许可证

- **应用代码（Kotlin/Java/Gradle）**：Apache 2.0
- **原生库（.so）**：从 CrispASR 和 MNN 构建，适用其各自的许可证
- **CosyVoice3 模型**：Apache 2.0（[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)）
- **MNN**：Apache 2.0（[alibaba/MNN](https://github.com/alibaba/MNN)）

---

## 致谢

- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — 阿里巴巴 CosyVoice 语音合成模型
- [alibaba/MNN](https://github.com/alibaba/MNN) — 阿里巴巴 MNN 推理引擎
- [CrispASR/mnn-cosyvoice-jni](https://github.com/crisp-oss/mnn-cosyvoice-jni) — MNN CosyVoice JNI 绑定
- [gedoor/Legado](https://github.com/gedoor/Legado) — 阅读 Archive，本项目的来源
- 所有在 [Issues](https://github.com/Nian27/Fun-CosyVoice/issues) 中反馈问题、提交代码的贡献者

---

> **遇到问题？** → [提交 Issue](https://github.com/Nian27/Fun-CosyVoice/issues/new)
>
> 提交时请附上：
> - 手机型号 / Android 版本
> - Logcat（`adb logcat -s "CosyVoice*" "*:E"`）
> - 如果是合成问题，提供 `run-{timestamp}/` 目录下的所有日志文件
