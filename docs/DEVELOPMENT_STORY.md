# Fun-CosyVoice 开发全流程记录

> 从 Legado Archive 提取 CosyVoice3 本地 TTS 为独立 App 的完整过程。
>
> 记录每一步做什么、为什么做、改了什么文件，方便后续开发者理解代码演变和定位问题。

---

## 目录

1. [背景：CosyVoice3 + Android 的挑战](#1-背景cosyvoice3--android-的挑战)
2. [第一阶段：CrispASR/ggml + Vulkan 路线（失败）](#2-第一阶段crispasrggml--vulkan-路线失败)
3. [第二阶段：转向 MNN/OpenCL](#3-第二阶段转向-mnnopencl)
4. [第三阶段：Flow 蒸馏（关键突破）](#4-第三阶段flow-蒸馏关键突破)
5. [第四阶段：MNN 手机端集成](#5-第四阶段mnn-手机端集成)
6. [第五阶段：独立 App 化](#6-第五阶段独立-app-化)
7. [研究脚本说明](#7-研究脚本说明)
8. [性能数据汇总](#8-性能数据汇总)
9. [Bug 修复记录](#9-bug-修复记录)
10. [后续可能的改进方向](#10-后续可能的改进方向)

---

## 1. 背景：CosyVoice3 + Android 的挑战

### CosyVoice3 模型结构

CosyVoice3（0.5B 参数）合成一条语音需要 4 个模块串联：

```
文字 → [LLM] → speech tokens → [Conditioner] → mel 特征 → [Flow] → mel 频谱 → [HiFT] → WAV
```

| 模块 | 参数 | 计算强度 | 手机推理难度 |
|------|------|----------|-------------|
| LLM | ~370M | 自回归，逐 token 生成 | 高（内存带宽瓶颈） |
| Conditioner | ~30M | 单次前馈 | 低 |
| Flow | ~330M | 2 步扩散，大矩阵乘 | 极高（需 GPU） |
| HiFT | ~85M | 1 次前馈 | 中 |

### 为什么难

- **模型太大**：完整 FP32 模型 ~6 GB，必须量化到 FP16 才能在手机上跑（~1.3 GB）
- **GPU 依赖**：Flow 的 DiT 结构在 CPU 上极慢（RTF > 5），必须用 GPU
- **OpenCL 碎片化**：Adreno / Mali / PowerVR 的 OpenCL 实现不同
- **Android linker namespace**：子进程不能继承 OpenCL 库

---

## 2. 第一阶段：CrispASR/ggml + Vulkan 路线（失败）

### 2026-07-14 尝试

最初选择 **CrispASR v0.8.10**（基于 ggml + Vulkan）作为推理后端，因为：
- CrispASR 提供了 CosyVoice3 的 C 接口
- ggml 是纯 CPU 推理框架，Vulkan 可加速
- 预编译包约 39 MB，解压后 ~150 MB

### 遇到的问题

1. **Adreno Vulkan DeviceLost**：在 `lm_ar_decode`（LLM 自回归解码）阶段报 `vk::Queue::submit: ErrorDeviceLost`，LLM 用 Vulkan 跑不稳定
2. **FP16 噪声**：关闭 Vulkan FP16 虽可避开 DeviceLost，但输出是刺耳噪声
3. **逐模块排查**：把 LLM/Flow/HiFT 逐个从 Vulkan 迁到 CPU，发现每个阶段都有兼容问题
4. **RTF 不可用**：纯 CPU 模式下 33 字合成 ~37.6 秒，RTF ~5.11

### 结论

CrispASR/ggml + Vulkan 在高通 Adreno GPU 上不可靠，Miq 路线废弃。

---

## 3. 第二阶段：转向 MNN/OpenCL

### 2026-07-14 决策

根据用户建议从 CrispASR/ggml 转向 **阿里 MNN** 推理引擎，用 OpenCL 后端做 GPU 加速。

### 模型转换流水线

```
PyTorch (.pt)  →  ONNX (.onnx)  →  MNN (.mnn)  → 手机推理
```

Stage 1 脚本：`research/mnn-cosyvoice3/build-stage1-android.ps1`

### 关键问题及修复

#### FP16 Mish 溢出（`stabilize_softplus_onnx.py`）

Softplus 激活函数计算 `exp(x)`，在 FP16 下容易溢出。重写为稳定形式：
```python
max(x, 0) + log(1 + exp(-abs(x)))
```

#### GELU 立方溢出（`stabilize_gelu_onnx.py`）

tanh-GELU 子图的立方分支在 FP16 下溢出，裁剪到 `[-10, 10]`。

#### Attention score 溢出（`stabilize_attention_onnx.py`）

实验性修复，修改后误差太大，最终拒绝 FP16 Normal 精度路径，强制使用 OpenCL **High** 精度。

#### MNN 外部权重

模型权重保存为 `.mnn.weight` 外部文件（~633 MB），不嵌入图文件。手机端用 `createFromFile()` 加载。

### MNN vs ONNX 数值验证

```
Sequence 256, OpenCL High: 最大绝对误差 ≈ 0.00326
Sequence 414, OpenCL High: 最大绝对误差 ≈ 0.00373
```

---

## 4. 第三阶段：Flow 蒸馏（关键突破）

### 2026-07-15 瓶颈

原始 Flow 使用 Euler solver **10 步**，真机实测单步 OpenCL High ~1.838 秒（sequence 414），10 步 ≈ 18 秒，完全不可用。必须蒸馏。

### 步骤 1：CFG 单分支蒸馏

原版 Flow 使用 Classifier-Free Guidance (CFG)，每次运行需要 2 个分支（cond + uncond），batch=2。

**目标**：训练单分支学生拟合教师 `1.7 * v_cond - 0.7 * v_uncond`

- 训练数据：24 条真实 CosyVoice 推理轨迹
- 训练参数：最后 8 个 DiT block + 输出层，共 119M / 331M 参数
- RTX 4060 Ti 显存峰值 ~3.27 GB

**结果**：
- 10 步 mel 余弦 0.99920，SNR 27.93 dB
- 直接降到 4/2 步质量差（余弦 0.99489/0.98172，SNR 19.82/14.13 dB）
- 单分支 batch=1 MNN 真机：seq414 OpenCL High 从 1.838 秒降到 **0.912 秒**

### 步骤 2：两步宏轨迹蒸馏

**时间点**：`0 → 0.2928932188 → 1`（按最优分布）

**方法**：学习密集 10 步教师轨迹两个端点间的**平均速度**

**数据**：20 条训练 + 4 条验证（同一 zero-shot 音色）

**结果**：
| 指标 | 值 |
|------|-----|
| Mel 余弦（验证集） | 0.99882 |
| Mel SNR（验证集） | 26.26 dB |
| 未见句 Mel 余弦 | 0.99843 |
| 未见句 Mel SNR | 24.99 dB |

### 步骤 3：真机验证

| 配置 | 单步耗时 | 两步总计 |
|------|---------|---------|
| seq516 OpenCL High | 1.187 秒 | **2.374 秒** |
| MNN vs ONNX 最大误差 | 0.00551 | — |

### 蒸馏结论

- ✅ 两步蒸馏 Flow 通过工程门槛
- ❌ 未通过产品门槛（100+ 句、多未见音色、盲听/CER/CAMPPlus/响度）
- OpenCL Normal/Low 输出非有限值，**禁止使用**

---

## 5. 第四阶段：MNN 手机端集成

### 2026-07-15 真机全流水线

手机端推理架构：

```
LLM (CPU Low / 4线程) → Conditioner (子进程) → Flow (OpenCL High) → HiFT (CPU High / 6线程)
```

### 子进程 → JNI 迁移

最初用 ProcessBuilder 启动原生可执行文件，但：

1. **子进程不能继承 OpenCL**：App linker namespace 不传递 `libOpenCL.so`
2. **加载慢**：每次合成要重新加载模型

**修复**：将 LLM/Flow/HiFT 全部迁移到 App 进程内 JNI：

| JNI 库 | 源文件 | 功能 |
|--------|--------|------|
| `libcosy_llm_jni.so` | CosyVoiceLlmNative.kt | LLM 常驻，每次 reset() 防串句 |
| `libcosy_flow_jni.so` | CosyVoiceFlowNative.kt | Flow，CPU/OpenCL 双后端 |
| `libcosy_hift_jni.so` | CosyVoiceHiFTNative.kt | HiFT core，CPU/OpenCL 双后端 |
| `libcosy_conditioner_exec.so` | — | Conditioner（仍为子进程） |
| `libcosy_enrollment_jni.so` | CosyVoiceEnrollmentNative.kt | 音色注册（可选） |

### Flow 桶机制

Flow 模型是动态形状的，首次未见长度会触发 ~12-17 秒 GPU kernel 编译。

**解决方案**：预定义桶大小 `{512, 768, 1024, 1280, 1536, 2048}`，补零到最近的桶。

**验证**：seq452 补零到 512，目标区域输出与精确 seq452 逐值一致。

### GPU 调度优化（2026-07-20 SM8850）

在荣耀 Magic8 Pro (SM8850) 上对比三种 OpenCL 内存模式：

| Mode | Flow 耗时 |
|------|-----------|
| Image(132) | ~3.75 秒 |
| 自动(4) | ~2.5 秒 |
| **Buffer(68)** | **~0.81 秒** |

**结论**：Flow 默认使用 Buffer(68)，HiFT 维持 CPU/6 线程。

### GPU 编译缓存

```kotlin
val cache = File(gpuCacheDir, "flow-fp16-${backend}-${precision}-mode${mode}.cache")
```

不同 backend/mode 的 kernel/tuning cache 持久化，避免每朗读 session 重编译。

### 端到端性能演进

| 日期 | RTF | 说明 |
|------|-----|------|
| 07-14 | 5.11 | CrispASR 纯 CPU |
| 07-15 | 1.27 | MNN LLM CPU + Flow OpenCL + HiFT CPU（推理耗时） |
| 07-15 | 1.47 | 热态连续压测 |
| 07-20 | 0.89 | Flow Buffer(68) + HiFT CPU/6 + 持久 JNI |
| 07-20 | **0.79** | 最优热态 |

---

## 6. 第五阶段：独立 App 化

### 从 Legado 剥离

原代码在 `阅读 Archive` 的 `cosytest` 分支中，与朗读流程耦合严重。

**剥离内容**：

| 组件 | Legado 位置 | 独立 App 位置 |
|------|-------------|-------------|
| JNI 接口 | `CosyVoiceNative.kt`（混合文件） | 4 个独立文件（Llm/Flow/HiFT/Enroll） |
| 合成管线 | `CosyVoiceRuntime.kt` | `CosyVoiceRuntime.kt`（去耦合） |
| 音色管理 | `CosyVoiceModelStore.kt` | `CosyVoiceStore.kt` |
| 模型管理 | 同上 | 同上（+ ZIP 导入/导出） |
| UI | `CosyVoiceManagerActivity.kt`（XML） | `MainActivity.kt`（Compose） |

### 新增功能

- **ZIP 模型导入/导出**：17 文件 SHA-256 校验，原子写入
- **音色档案管理**：多音色切换、导入/导出、实时版生成
- **手机创建音色**：Enrollment JNI + MediaCodec 解码
- **Instruct2 指令演绎**：四川话、广东话、情感控制等
- **GPU 调度 UI**：CPU/OpenCL 切换、Mode 选择

### Bug 修复

详见第 9 节。

---

## 7. 研究脚本说明

`research/mnn-cosyvoice3/` 目录包含整个移植过程的可复现脚本：

### 目录结构

```
research/mnn-cosyvoice3/
├── *.ps1                     # PowerShell 构建/运行脚本
├── scripts/*.py              # Python：蒸馏、模型导出、数值验证
├── data/*.txt                # 输入文本：蒸馏训练/验证数据
├── cases/*/                  # ONNX/MNN 测试用例（JSON 格式）
├── STAGE3_FEASIBILITY.md     # 蒸馏可行性报告
├── VOICE_ENROLLMENT*.md      # 音色注册方案
└── *.json                    # Stage manifest（文件清单和 SHA-256）
```

### 主要脚本

| 脚本 | 作用 |
|------|------|
| `build-stage1-android.ps1` | 编译 Flow MNN Android benchmark |
| `build-stage2-android.ps1` | 编译 HiFT MNN Android benchmark |
| `build-cosytest-model-zip.ps1` | 构建 17 文件模型 ZIP |
| `build-enrollment-extension.ps1` | 构建音色注册扩展 ZIP |
| `run-phone-e2e.ps1` | 手机端一键全流水线（LLM→Flow→HiFT→WAV） |
| `run-phone-concurrent-benchmark.ps1` | 三模块并发真机基准 |
| `run-stage3-cfg-distill.ps1` | 完整的 CFG + 两步蒸馏流水线 |
| **scripts/capture_cfg_trajectory.py** | 捕获 CFG 教师推理轨迹 |
| **scripts/train_cfg_student.py** | 训练 CFG 学生 / 两步轨迹 |
| **scripts/build_two_step_cache.py** | 构建两步宏轨迹缓存 |
| **scripts/export_flow_estimator_variant.py** | 导出单分支 batch=1 Flow |
| **scripts/stabilize_softplus_onnx.py** | ONNX Softplus FP16 稳定性修复 |
| **scripts/stabilize_gelu_onnx.py** | ONNX GELU FP16 稳定性修复 |
| **scripts/export_hift_split.py** | HiFT 分割导出（F0 + Core + DSP） |
| **scripts/build_voice_profile.py** | 音色档案构建 |
| **scripts/evaluate_cfg_student_audio.py** | 蒸馏音频质量评估 |

---

## 8. 性能数据汇总

### 端到端（荣耀 Magic8 Pro / SM8850 / Adreno 840 / 12 GB RAM）

| 配置 | 冷启动 | 热态 |
|------|--------|------|
| **RTF** | ~1.7 | **0.79 - 0.96** |
| LLM CPU/4t | ~1.6 秒 | ~1.6 秒 |
| Flow GPU Buffer(68) | ~3.4 秒（形状编译） | 0.8 - 1.1 秒 |
| HiFT CPU/6t | ~1.9 秒 | 1.4 - 2.0 秒 |
| 进程内存 | ~947 MB | ~947 MB |

### 蒸馏提升对比

| Flow 配置 | seq414 单步 | 两步总计 | 备注 |
|-----------|-------------|---------|------|
| 原始 CFG batch=2 | 1.838 秒 | 18.38 秒（10 步） | 不可用 |
| 单分支学生 batch=1 | 0.912 秒 | 9.12 秒（10 步） | 单步快但步数多 |
| 两步蒸馏 batch=1 | **1.187 秒** | **2.374 秒** | ✅ 可用 |

### 热降频影响

连续合成后 SoC 升温，性能下降：
- 冷机峰值吞吐 RTF：**0.63**（三句并发）
- 热机连续 RTF：**1.29**（降压限频）
- 当前不支持缓存水位暂停或自动背压

---

## 9. Bug 修复记录

### #1: CrispASR/Vulkan DeviceLost（已弃用路线）

**方案**：放弃 CrispASR/ggml，全面迁移到 MNN/OpenCL

### #2: MNN FP16 OpenCL 非有限输出

**修复**：强制 OpenCL **High** 精度，禁用 Normal/Low

### #3: OpenCL 子进程不能继承

**修复**：所有 GPU 阶段改为 App 进程内 JNI

### #4: 动态形状 GPU kernel 重编译

**修复**：Flow 桶机制 + HiFT mel 帧桶，首次预热

### #5: MNN LLM 输出全零

**根因**：缺少 `llm_config.json`（456 bytes）

**修复**：模型包包含该文件并 SHA-256 校验

### #6: MNN 外部输入别名覆盖

**根因**：MNN 动态图复用输入缓冲区存储，多次 runSession() 不重写输入会改变结果

**修复**：每次 runSession() 前重写所有外部输入

### #7: 首次打开 App 空指针

**修复**：`CosyVoiceStore.selectVoiceProfile()` 加 `modelStatus().ready` 保护

### #8: libOpenCL.so required=true（cosytest → 独立 App）

**修复**：独立 App 的 Manifest 不加 `<uses-native-library>`，运行时检测 OpenCL 可用性

### #9: Flow 和 HiFT OpenCL 同时接入时冲突

**根因**：共享 MNN Session 状态

**修复**：`setCacheFile/updateCacheFile` 按模块、精度和 mode 独立

### #10: HiFT CPU 线程数错误

**根因**：CPU HiFT core 误用 `hiftGpuMode=4` 而非 `threads=6`

**修复**：CPU 路径显式传 6 线程

---

## 10. 后续可能的改进方向

### 高优先级

1. **非高通设备验证**：麒麟/天玑的 OpenCL 兼容性未知
2. **单 Native Runtime**：合并 LLM/Flow/HiFT/Conditioner 为单一 JNI 共享内存
3. **流水线重叠**：LLM/分块 Flow/分块 HiFT 流式处理

### 中优先级

4. **模型分片下载**：支持断点续传
5. **更多蒸馏数据**：100+ 句、多未见音色验证
6. **CAMPPlus 音色余弦门槛**：客观指标把关

### 低优先级

7. **MNN QNN/NPU 后端**：骁龙 HTP/NPU 加速
8. **音色市场**：用户共享音色档案
9. **English UI**

---

> **最后更新**：2026-07-21
>
> **完整研究目录**：`research/mnn-cosyvoice3/`（包含所有可复现脚本）
>
> **反馈**：[GitHub Issues](https://github.com/Nian27/Fun-CosyVoice/issues)
