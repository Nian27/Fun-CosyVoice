# CosyVoice3 音色注册与指令控制实施方案

## 目标

- 用户选择一段 MP3/WAV，经过一次较慢的本地处理，创建可复用音色。
- 创建后的音色进入阅读 App 的发言人库，可分配给旁白或角色。
- 每句朗读不再解析参考音频，只加载音色档案，保持热态 RTF 小于 1。
- 支持语言、方言、情绪、风格、语速、音量倾向和自由指令。

## 两条独立管线

### 音色注册（低频、允许慢）

1. 使用 Android SAF 选择 MP3/WAV。
2. 通过 `MediaExtractor`/`MediaCodec` 解码为单声道 float PCM。
3. 去除首尾静音，检查单人、无配乐、无削波；较长干净语音可用于 speaker embedding，但自动挑选最稳定的 3-5 秒作为实时 prompt。
4. 分别重采样到 16 kHz 和 24 kHz。
5. 16 kHz Whisper 128-bin Mel -> `speech_tokenizer_v3` -> speech tokens。
6. 16 kHz Kaldi 80-bin fbank、均值归一化 -> CAMPPlus -> 192 维 speaker embedding。
7. 24 kHz 80-bin Mel -> prompt speech feature。
8. 对齐长度：`prompt_frames = 2 * prompt_tokens`，多余部分裁掉；实时 prompt 建议限制在 125 tokens/250 frames 以内，避免每句话的 Flow 上下文随参考音频过度增长。
9. 用户输入参考音频的准确文本；后续可增加可选本地 ASR 自动转写，但必须允许人工校对。
10. 原子写入音色档案，试听通过后才加入发言人库。

### 实时合成（高频、必须快）

1. 根据角色取得 `voiceProfileId`，读取已缓存的音色档案。
2. 规范化结构化指令，形成稳定的 instruction text。
3. LLM 生成目标 speech tokens。
4. Flow Buffer(68) 生成 Mel。
5. HiFT CPU High/6 生成 24 kHz WAV。
6. 音频缓存键包含模型版本、音色档案哈希、指令、速度和正文。

注册完成后，实时链路不得再次运行 MP3 解码、speech tokenizer、CAMPPlus 或 prompt Mel 提取。

## 音色档案格式

建议每个音色独立目录：

```text
voices/<voice-id>/
  profile.json
  prompt-text-tokens.i32
  prompt-speech-tokens.i32
  prompt-feat.f32
  speaker-embedding.f32
  runtime-prompt-cond.f32
  runtime-flow-spks.f32
  source-audio.m4a        # 可选，供重新注册或试听
```

`profile.json` 至少包含：

```json
{
  "schemaVersion": 1,
  "modelId": "Fun-CosyVoice3-0.5B-2512-RL-distilled",
  "voiceId": "uuid",
  "displayName": "自定义音色",
  "promptText": "参考音频的准确文本",
  "promptTokenCount": 87,
  "promptFrameCount": 174,
  "speakerEmbeddingSize": 192,
  "profileHash": "sha256",
  "createdAt": 0
}
```

当前代码中的 `PROMPT_TOKENS=87`、`PROMPT_FRAMES=174` 和固定 prompt 文件必须改成从档案读取。

`prompt-feat` 和 192 维 `speaker-embedding` 是可重新编译的原始档案；`runtime-prompt-cond` 与 80 维 `runtime-flow-spks` 是当前蒸馏 Flow 可直接读取的运行时档案。当前基准音色的运行时条件约 56 KiB，每句不应重复执行 speaker projection。模型版本变化时从原始档案重新生成运行时档案。

## 指令控制

UI 使用结构化字段，不让每个调用方自行拼提示词：

- 语言：普通话、英语、日语等。
- 方言：四川话、粤语、东北话等。
- 情绪：平静、开心、悲伤、愤怒、紧张等。
- 风格：旁白、聊天、严肃、温柔、激动等。
- 生成语速：模型控制，影响表达与停顿。
- 表达音量：如轻声、响亮，属于演绎风格。
- 自由指令：高级用户补充。

最终响度仍由阅读 App 的 LUFS 归一化和播放音量处理，不能只依赖“响亮/轻声”指令。模型指令控制表演方式，播放器控制可重复的实际音量。

CosyVoice3 `instruct2` 的 LLM 输入只使用 instruction text，不再把参考 speech tokens 放进 LLM；Flow 仍使用音色档案中的 prompt speech tokens、prompt feat 和 speaker embedding。因此运行时需要显式区分：

- `ZERO_SHOT`：参考文本 token + LLM prompt speech token。
- `INSTRUCT2`：指令文本 token，不传 LLM prompt speech token。

## 模型包拆分

## 指令模式真机结果（2026-07-20）

- `cosytest` 已加入“普通复刻/指令演绎”切换、官方指令预设和自由指令输入。
- `ZERO_SHOT` 保持原链路：提示文本来自音色 profile，并向 LLM 附加参考 speech tokens。BKQ_AN90 实测报告为 `promptSpeechTokensAppended=true`，生成 127 个 speech tokens、5.08 秒 WAV。
- `INSTRUCT2` 使用 `You are a helpful assistant. <instruction><|endofprompt|><text>`，不向 LLM 附加参考 speech tokens；Flow 仍读取同一音色 profile。四川话测试报告为 `promptSpeechTokensAppended=false`，生成 129 个 speech tokens、5.16 秒 WAV，热态总耗时 4.47 秒、RTF 0.87。
- 同文本“尽可能快”指令生成 77 个 speech tokens、3.08 秒 WAV，热态总耗时 2.47 秒、RTF 0.80；相比普通复刻时长明显缩短，证明速度指令已进入真实推理分支。
- 普通、四川话和快速三次输出的 speech-token 文件与 WAV SHA-256 均不同，PCM peak/RMS 正常，无 JNI/OpenCL 崩溃。
- 当前结果只证明指令链路和速度控制生效。四川话、情绪、风格等主观质量仍需人工听测和批量样本评估，不能标记为产品质量已通过。

### 基础实时包

保留当前约 1.3 GB 的 17 文件包，负责已注册音色的实时合成。

### 可选音色创建扩展

- `speech_tokenizer_v3.onnx` 当前约 925 MiB。
- `campplus.onnx` 当前约 27 MiB。
- 音频 Mel/fbank/重采样前端。
- 必要时单独导出的 prompt conditioner。

优先把两个 ONNX 转为 MNN，复用现有运行时；若 speech tokenizer 转换失败，再只为注册模块引入 ONNX Runtime Mobile。扩展模型不进 APK，可在线下载或导入 ZIP。音色注册完成后允许删除扩展模型，已生成的音色档案继续可用。

## 与阅读发言人管理集成

- 新增发言人来源类型 `COSYVOICE_PROFILE`，`engineValue` 保存 `voiceId`，不保存模型路径。
- 音色创建成功后直接写入发言人库，支持试听、重命名、删除和重新注册。
- 删除音色前二次确认；同时使引用该音色的角色进入“音色失效”状态，不静默换成其他声音。
- 旁白和主角的唯一性规则继续由现有角色音色分配层负责。

## 缓存键

```text
sha256(
  modelVersion + profileHash + mode + normalizedInstruction +
  modelSpeed + outputSampleRate + normalizedText
)
```

修改音色、指令或模型速度只失效相关缓存，不清空整本书。

## 实施顺序与验收

1. 动态音色档案：已完成。默认 profile 在 BKQ_AN90 真机首次原子迁移成功，三个运行时文件与模型源文件 SHA-256 一致，共享 noise 软链接正确；冷启动及两次热态完整 WAV 均成功，热态 RTF 为 0.94/0.79，profile 不会按句重建。
2. 指令模式：基础 `ZERO_SHOT/INSTRUCT2` 分支和测试 UI 已完成；下一步建立语言、方言、情绪批量测试矩阵并做人工听测。
3. CAMPPlus 手机端：对桌面 ONNX 输出做余弦一致性验证。
4. speech tokenizer 手机端：逐 token 对齐桌面输出。
5. MP3 注册 UI：进度、取消、错误诊断、试听和二次确认删除。
6. 接入发言人库和角色分配。
7. 至少 20 个未见音色、100 句验证音色相似度、CER、响度、削波、冷/热 RTF 和连续运行温升。

不得在第 3、4 步数值一致性通过前，把“选择 MP3”入口作为可用功能发布。

## 角色 AI 到朗读指令参数（2026-07-20）

- 语言、方言/口音和情绪不新增第二次 AI 请求，而是与角色身份一起由现有多角色 AI 批次返回。每个朗读 unit 固定包含 `languageCode`、`accentCode`、`emotionCode`。
- 默认值固定为 `zh/mandarin/neutral`。只有原文明确出现外语、方言/口音或情绪证据时才允许覆盖，避免模型为普通旁白和台词随意增加表演风格。
- 支持 9 种语言：`zh/en/ja/ko/de/es/fr/it/ru`；支持 18 种中文方言/口音：普通话、广东话、闽南话、四川话、东北话、陕西话、上海话、天津话、山东话、宁夏话、甘肃话、贵州话、河南话、湖北话、湖南话、江西话、山西话、云南话；非中文统一使用 `standard` 口音。
- 当前结构化情绪为中性、开心、伤心、生气、害怕、惊讶、焦虑、激动、温柔、严肃。它与旧 HTTP TTS 的 `emotionName/emotionTag` 分开，避免把通用 `neutral` 误传给不兼容的 HTTP 接口。
- 三个字段进入角色缓存、角色分配明细、语音计划、路由签名和音频缓存键。字段变化只使对应片段音频失效，不清空整章或整本书。
- `cosytest` 已提供 `ReadAloudPerformance -> Instruct2` 转换；正式 `app` 仍未接入 CosyVoice 合成器，现有 Edge/HTTP 引擎能否表现某种语言或方言取决于具体发言人和接口能力。
