# 项目记忆

## 2026-07-14

- 用户要求 CosyVoice3 完全在 Android 设备本地运行，不依赖电脑、局域网或远程推理服务。
- 阅读项目路径：`E:\AndroidStudioProjects\legado-archive-v3-3.26.06071119`。
- CosyVoice 源码路径：`E:\AndroidStudioProjects\CosyVoice-main`。
- 当前调查方向：使用 Android arm64 原生 C++/GGUF 推理运行时，模型独立下载或导入，不放入 APK。
- 已下载并检查 CrispASR v0.8.10 Android arm64 预编译包，位置为 `research\crispasr-v0.8.10`；压缩包约 39 MB，原生库解压后约 150 MB。
- 已确认 C API 支持 CosyVoice3 合成、预置音色和“参考 WAV + 精确文本”克隆；最小克隆模型组合约 897 MB，未下载模型权重。
- 2026-07-14 检查时 ADB 没有连接设备，因此尚未进行手机端内存、速度和稳定性验证。
- CosyVoice3 独立测试包使用 `io.legado.app.cosytest.debug`，与正式阅读包并存；模型保留在应用外部数据目录，不打入 APK。
- 设备端全 CPU 基准为 33 字合成约 37.6 秒、音频约 7.36 秒，RTF 约 5.11，无法满足边读边合成。
- Adreno Vulkan 全 GPU 路径已用分阶段日志定位到 `lm_ar_decode` 的 `vk::Queue::submit: ErrorDeviceLost`，问题位于自回归 LLM 单 Token 解码阶段，不是播放器或手机算力不足。
- 禁用 Vulkan FP16 虽可避开 DeviceLost，但输出为满幅异常 PCM 和刺耳噪声；该路线禁止继续使用。测试包已增加 PCM 健康检查，在播放前拦截非有限值、过高峰值和超范围样本。
- 2026-07-14 真机分阶段验证：LLM 权重与 KV Cache 转 CPU 后成功越过 `lm_ar_decode`，但 Flow 首阶段 `pre_la+interleave` 在 Vulkan 上 `ErrorDeviceLost`；再将 Flow 转 CPU 后成功越过 Flow，HiFT 在 `hift_vocoder` 阶段仍触发 `ErrorDeviceLost`。失败前约耗时 55.2 秒且尚未产出音频，因此 CrispASR/ggml 的 CPU/Vulkan 混合路线在当前 Adreno 设备上不具备朗读可用性，不再继续叠加 Vulkan 兼容开关。
- 用户提供的 MNN/OpenCL 建议与当前 CrispASR + ggml/Vulkan 不是同一运行时；迁移 MNN 需要模型转换和推理层重写，不能通过升级依赖或修改线程参数直接生效。
- 本地 `CosyVoice-main` 仅提供 Flow estimator 的 ONNX 导出以及 Python/Triton/TensorRT 服务端运行时，没有 Android/MNN 或完整移动端导出链；MNN/OpenCL 方案必须作为独立移植项目，逐模块完成 LLM、Flow、HiFT 的模型导出、数值一致性和手机端基准验证。
- MNN Flow 真机基准的 `11.52 秒`是 sequence-256 单次 estimator 稳态中位数 `1.152 秒`乘官方 Euler `10` 步得到的计算估算，不是完整流水线直接生成 WAV 的耗时。CosyVoice3 每个 mel 帧对应 480 个 24 kHz 采样点，因此 256 帧总上下文为 5.12 秒，按总上下文算 Flow RTF 约 2.25；若其中包含约 2 秒提示音频，净新生成约 3.12 秒，对应有效 Flow RTF 约 3.69。
- `Voine/Bert-VITS2-MNN` 是更接近可直接集成的 Android 离线 TTS 候选：仓库提供 MNN 全流程、中文/日文/英文预处理和可发布 AAR，公开的骁龙 888 参考数据为 5.20 秒音频端到端约 1.856 秒（RTF 0.357）。它适合固定角色音色和小说预合成，但新增音色通常需要训练 Bert-VITS2 2.3 模型并转换，不能替代 CosyVoice3 的参考音频零样本克隆。代码许可与具体底模、角色训练数据许可必须分别核对，未经授权的游戏角色音色不能随 App 分发。
- 用户已决定放弃本地 TTS，停止 CosyVoice3/Bert-VITS2 的正式接入和后续性能移植；主项目现有 CosyVoice 代码保持在独立 `cosytest` flavor，不进入正式 `app` 包。后续优化集中于在线 TTS 的可靠路由、缓存/播放状态机、切章一致性、响度稳定和独立朗读页性能。
- 2026-07-15 用户提供新的手机端蒸馏方案后，恢复了一个隔离的 CosyVoice3 可行性实验，但仍未恢复正式 App 接入；阅读主项目和 APK 未改动。
- Stage 3 使用官方 `Fun-CosyVoice3-0.5B-2512` revision `29e01c4e8d000f4bcd70751be16fa94bf3d85a18`，`llm.rl.pt` 通过 NTFS 硬链接作为加载器要求的 `llm.pt`。模型位于 `E:\AndroidStudioProjects\cosyvoice3-distill-lab\Fun-CosyVoice3-0.5B-2512-RL`，大模型不进入 APK。
- 桌面环境复用 `D:\anaconda3\envs\pytorch` 的 Python 3.9.19、PyTorch 2.4.1、CUDA 11.8；附加依赖位于 `E:\AndroidStudioProjects\cosyvoice3-distill-lab\python-deps`，不得用 `pip --target` 自动解析依赖，否则会重复下载 CPU Torch 2.8。环境脚本必须使用 `--no-deps` 并先验证现有 CUDA Torch。
- RTX 4060 Ti 真机桌面推理验证：普通中文 6.08 秒输出，10/4/2 步端到端分别 6.02/5.93/7.28 秒，RTF 0.990/0.976/1.198；CAMPPlus 相对 10 步音色余弦 1.000/0.976/0.916，Whisper Base 三档 CER 均为 0（繁转简后）。绕口令 9.36 秒输出，RTF 1.206/0.978/0.894，音色余弦 1.000/0.990/0.930。
- 直接降步会系统性抬高响度：两组样本中 4 步相对 10 步约 +1.93 到 +2.41 dB，2 步约 +5.05 到 +5.36 dB。结论是只进入 10->4 步 Flow 真蒸馏 PoC；未经蒸馏的 2 步方案拒绝，不能直接接入阅读 App。LLM 0.3B 学生须等 4 步 Flow 通过至少 100 句、多音色、盲听/CER/音色/响度/手机 RTF 门槛后再开始。
- 2026-07-15 MNN LLM 真机验证：合并后的 CosyVoice3 LLM 使用 Transformer INT4、lm_head INT8，总模型约 638 MB；speech EOS `158486` 已写入 MNN tokenizer。真实输入包含 87 个参考语音 token，完整 prompt 为 134 token。
- 已为 MNN 增加隔离的 `cosyvoice_ras` 采样器，只允许 6561 个正常语音 token 和 EOS，并按原版 top-p 0.8、top-k 25、10-token 窗口执行重复感知重采样。三次 CPU 真机运行均生成 118-125 个目标 token 后正常 EOS，未出现越界 token。
- LLM 真机最优为 CPU Low、4 线程，RAS 中位速度约 84.99 token/s；OpenCL Low 热缓存约 68.38 token/s，且冷启动 prefill 约 10.09 秒，因此 LLM 不上 GPU。Flow 使用 OpenCL High，HiFT 使用 CPU High。
- 真实约 120 个目标 token 时，Flow estimator 序列长度约为 `2 * (87 + 120) = 414`。手机 OpenCL High 单次 estimator 稳态中位数约 1.838 秒，数值对 ONNX 最大绝对误差约 0.00326；4 步约 7.35 秒，而目标语音约 4.8 秒，Flow 单模块 RTF 已约 1.53。当前完整模型无法达到端到端 RTF < 1，下一步必须做 Flow 结构蒸馏/裁剪，不能靠切换后端、线程数或未经蒸馏的降步解决。
- MNN 外部权重文件名必须与图文件基名一致；缺失对应 `.weight` 时运行时可能打印 `Can't open file` 后仍返回全零“成功”输出，所有基准必须同时检查 finite、RMS 和 ONNX 数值误差。
- 2026-07-15 CFG 单分支性能门槛：从同一 Flow 权重重新导出固定 batch=1 ONNX/MNN 图，外部权重 SHA-256 与 batch=2 图完全一致。seq414/OpenCL High 稳态中位数从 batch=2 的 1.838 秒降到 batch=1 的 0.912 秒，FLOPs 从 172625M 降到 86304M，MNN 对 ONNX 最大绝对误差约 0.00373。
- batch=1 图只是条件分支，不能直接替换 CFG 教师，否则 guided velocity 和音质会变化。推荐训练路径是先做 CFG distillation，让单分支学生拟合教师 `(1+cfg)*v_cond-cfg*v_uncond`，再做 2 步轨迹/一致性蒸馏。按真机数据，两次单分支 Flow 约 1.82 秒；配合 LLM/Flow/HiFT 流式重叠，有机会达到稳态 RTF < 1，但顺序单句总延迟仍接近门槛。
- 2026-07-15 已完成 CFG 单分支和两步宏轨迹蒸馏工程 PoC。使用 24 条真实 CosyVoice 推理轨迹、单一 zero-shot 提示音，20 条训练、4 条整句验证；训练 Flow 最后 8 个 DiT block 和输出层，共 119,728,208 / 331,142,224 参数，RTX 4060 Ti 峰值显存约 3.27 GB。
- CFG 教师目标必须严格使用 `1.7*v_cond-0.7*v_uncond`。24 句 CFG 学生在额外未见句上 10 步 mel 余弦 0.99920、SNR 27.93 dB；未经轨迹蒸馏直接降到 4/2 步仅为 0.99489/0.98172 和 19.82/14.13 dB，因此直接两步仍禁止。
- 两步宏轨迹时间点为 `0 -> 0.2928932188 -> 1`，每步学习密集 10 步教师轨迹两个端点之间的平均速度。最终两步学生在额外未见句上 mel 余弦 0.99843、SNR 24.99 dB；4 条整句验证的生成区域均值为 0.99882、26.26 dB。
- 最终两步 batch=1 学生 MNN 在 Honor BKQ_AN90、seq516、OpenCL High 单次 estimator 稳态中位数 1.187 秒，两次约 2.374 秒；MNN 对 ONNX 最大绝对误差 0.00551。OpenCL Normal/Low 虽约 0.633 秒/次但输出非有限值，禁止使用，High 精度是硬约束。
- 当前只通过 Flow 工程门槛，尚未通过至少 100 句、多未见音色、盲听、CER、CAMPPlus、响度/削波和 Android 端到端 RTF 门槛；不得把模块估算写成手机完整合成 RTF 已通过，也尚未接入阅读 App。
- 2026-07-15 已完成隔离的 Android 文本到 WAV 研究链路：MNN LLM CPU Low/4 线程 -> 动态 Flow conditioner -> 两步蒸馏 Flow OpenCL High -> HiFT F0/core CPU High/4 线程 -> 原生谐波源、STFT/iSTFT 和 WAV。CUDA 只用于电脑生成数值参考，手机不使用 CUDA。
- 固定真实句 seq516 的手机两步 Flow 中位约 2.405 秒，最终 mel 对桌面参考最大绝对误差 0.01099；342 帧 HiFT 约 2.711 秒，F0/source-STFT/PCM 最大误差分别约 0.00229/0.00174/0.01614，6.84 秒 WAV 有限且无削波。
- 手机 LLM 必须输入完整 `prompt_text + <|endofprompt|> + target_text` 并附加 87 个提示语音 token；只传目标文本会立即 EOS。Flow conditioner 的 MNN token 输入是 int32，按 ONNX int64 写入会造成最大约 6.44 的错误，运行时必须读取张量真实位宽。
- HiFT 原生声源必须在 mel 帧率累积相位后乘 480 再扩展到采样率；遗漏 480 会产生刺耳噪声。无测试参考时使用确定性 xorshift32 声源噪声也能得到 CER 0、无削波的可识别音频，不再依赖电脑预生成 `source-noise-table.bin`。
- Flow 两步之间不能复用静态外部输入；MNN 动态图会别名并覆盖 `mask/mu/spks/cond` 存储，省略第二次上传会把最终 mel 最大误差扩大到约 12.1。该优化已回退。HiFT 维持 CPU High/4 线程，持续负载下 6/8 线程更慢。
- `research\mnn-cosyvoice3\run-phone-e2e.ps1` 可从任意输入文本一键执行手机 LLM->Flow->HiFT，结果保存到 `results\phone-e2e\<timestamp>`，最新 WAV 复制到手机 `/sdcard/Download/cosyvoice3-mnn-latest.wav`。
- 完整流水线尚未达到产品门槛：一次 124-token/4.96 秒音频测试约 5.76 秒且 Whisper CER 0；连续压测后 LLM 从约 85 token/s 降到 42.6 token/s，一键运行生成 5.56 秒音频耗时 8.18 秒、顺序 RTF 1.47、CER 4.76%。下一步必须实现持久 Session 和 LLM/分块 Flow/分块 HiFT 流式重叠，并通过 100 句、多音色、热态 RTF/CER/CAMPPlus/响度门槛后才能接入 `cosytest`，当前不修改阅读 App。
- 2026-07-15 已修正旧端到端脚本的计时口径：3.92 秒音频的模块推理为 4.97 秒（RTF 1.27），但真实冷启动墙钟为 20.04 秒（RTF 5.11），其中 Flow 加载和首次 OpenCL 形状编译为 12.73 秒。后续不得再把 inference-only RTF 当作用户首句等待时间。
- LLM、Flow、HiFT 三段常驻验证完成。三句共 14.8 秒音频的热态非重叠工作为 18.23 秒（RTF 1.23）；HiFT 常驻后同批耗时降至 5.53 秒。LLM 独立连续请求维持约 83-85 token/s，先前降速主要来自错误累计指标和重复加载，不是 KV 状态持续污染。
- Flow 必须固定桶运行：首次未见动态形状会触发约 12-17 秒 OpenCL 编译；真实 seq452 补零到 512 后，真实目标区域与精确 seq452 输出逐值一致。首选桶为 512/768/1024/1280/1536/2048，进入朗读前后台预热 512，禁止每句任意 resize。
- 自动并发真机基准已固化为 `research\mnn-cosyvoice3\run-phone-concurrent-benchmark.ps1`。Flow 512 桶预热后，LLM CPU、Flow Adreno OpenCL High、HiFT CPU 同时处理三句，冷机时 14.8 秒音频工作在 9.33 秒完成，峰值吞吐 RTF 0.63，三个进程均成功，PCM 峰值 0.58-0.82。该数据只证明冷机峰值跨句算力，不代表首句、持续热态或真实依赖队列已达标。
- FP16 Flow 候选把完整外部模型集降到 1,394,278,088 bytes（1.299 GiB），当前单句数值和 ASR 检查通过；Flow INT8 无速度收益且最终 mel 最大误差约 1.95，永久拒绝。模型必须安装后独立下载/导入，不进入 APK。
- APK 接入边界：先实现单一 MNN JNI 常驻运行时、依赖驱动句子队列、任务取消、固定桶预热和原子缓存输出，只接独立 `cosytest` flavor；旧 CrispASR/Vulkan 后端不能复用。当前手机资产只有一个预处理参考音色，任意录音音色克隆尚未实现，测试 UI 必须先隐藏该能力。正式阅读包仍需 100 句、多未见音色、盲听/CER/CAMPPlus/响度门槛后才能接入。
- 2026-07-15 完整自包含并发复测推翻“RTF 0.63 可持续”的过早结论：在 58 秒准备负载后立即运行，15.12 秒音频工作耗时 19.51 秒，RTF 1.29；Flow GPU 仍约 7.94 秒，LLM CPU/HiFT CPU 分别从冷机约 8.8/6.5 秒恶化到 18.93/14.29 秒。系统热状态为 1，皮肤温度接近轻度限频阈值。0.63 只能作为冷机峰值。按用户口径不采用缓存水位暂停或自动背压；连续朗读仅做跨句流水线，热降频负责监测与提示，不自动暂停合成。
- MNN GPU 调度已增加待真机验证的基建：Flow 与 HiFT core 可独立选择 OpenCL mode 4（Wide+自动内存）、68（Wide+Buffer）、132（Wide+Image）；HiFT F0/DSP 保持 CPU，只有大卷积 core 可迁 GPU。Flow/HiFT 同时接入 `setCacheFile/updateCacheFile`，kernel/tuning cache 按模块、精度和 mode 持久化，避免每个朗读 session 重编译 12-24 秒。代码已交叉编译通过，但因手机 ADB 断开，GPU HiFT、cache 复用和三种 mode 的真机结果尚未验证，禁止写成已通过。
- 2026-07-15 `cosytest` 已切换为应用私有目录内运行的 MNN 原生流水线。完整外部模型包固定为 17 个文件、1,399,078,584 bytes，必须包含 456-byte `llm_config.json`；遗漏该文件会触发 `Unable to open llm_config file`，随后出现 LLM Reshape 错误。
- 完整导入包位于 `E:\AndroidStudioProjects\cosyvoice3-mnn-mobile-fp16-complete.zip`，ZIP 为根目录扁平结构，含 17 个模型文件和 `manifest.json`，大小 1,399,083,563 bytes，SHA-256 `BAA7FB57C252DD231D0BDC5A758BAECDB43A4055B6E08DB49CE3CF0028D52EDF`。`research\mnn-cosyvoice3\build-cosytest-model-zip.ps1` 负责可重复生成并逐文件校验哈希。
- 手机 `io.legado.app.cosytest.debug` 已重新部署完整 17 文件。直接运行 App 内 LLM 程序成功，退出码 0、生成 speech token，不再出现配置缺失和 Reshape 错误。模型删除按钮必须经过明确二次确认，并说明删除后需重新导入约 1.3 GB 完整包。

## 2026-07-20 HiFT OpenCL JNI 修复

- Android App 通过 `ProcessBuilder` 启动的原生 ELF 子进程不能继承 `<uses-native-library libOpenCL.so>` 对应的 App linker namespace；即使设备提供 `libOpenCL.so`，子进程仍会报 `Can't create Runtime: OPENCL`。Flow/HiFT 等 GPU 阶段必须通过 App 进程内 JNI 执行，禁止再改回子进程 GPU。
- MNN 新增 `CosyVoiceHiFTJni.cpp` 和 `libcosy_hift_jni.so`；HiFT F0 与 DSP 继续使用 CPU，HiFT core 可选 Adreno OpenCL，并与 Flow 一样复用进程内 MNN Session 和磁盘 kernel cache。
- HiFT OpenCL 使用固定 mel 帧桶 `128/256/384/512/768/1024/1280/1536/2048`，输入补齐后按原始帧数裁剪 PCM。该设计避免每句不同 mel 长度都重新编译 GPU kernel；CPU 路径仍按精确帧数运行。
- 真机最终连续 GPU 验证：4.44 秒 WAV，总耗时 9.61 秒，Flow 3.34 秒、HiFT 3.04 秒；`melFrames=222`、`runtimeMelFrames=256`，Flow/HiFT `resizeMs=0`，PCM peak 0.686、RMS 0.069，无 OpenCL Runtime 错误或崩溃。当前顺序单句 RTF 仍约 2.16，尚未达到正式朗读实时门槛。
- 测试页阶段提示改为 suspend 回调并在 Main.immediate 顺序更新，修复合成已完成后旧的“LLM 正在生成”消息覆盖完成状态、造成假卡住的问题。
- 已安装测试包 `3.26.07201139-cosytestdebug`；APK 为 `E:\AndroidStudioProjects\legado_cosytest_3.26.07201139_460720113.apk`，68,187,634 bytes，SHA-256 `C098D1D236CDB53757103CC98BE1BE92DE96CBD84F7674872BA6FA157E2228C4`。APK 含 `libcosy_flow_jni.so` 与 `libcosy_hift_jni.so`，模型仍外置且完整保留。

## 2026-07-20 常驻 JNI、SM8850 调优与模型导出

- LLM 已从 `ProcessBuilder` 子进程迁移到 App 进程内 `libcosy_llm_jni.so`，按配置路径复用 MNN LLM；每次请求仍执行 `llm->reset()`，避免 KV/上下文串句。首次模型加载约 537.9 ms，第二次请求 `loadMs=0`。
- Flow、HiFT、LLM 三段均为常驻 JNI；旧的 `libcosy_llm_exec.so`、`libcosy_flow_exec.so`、`libcosy_hift_exec.so` 已从 APK 移除，Conditioner 暂时仍是子进程。退出测试页使用独立 IO release scope 释放 Session，真机 PSS 从约 946.7 MB 降到 268.9 MB，修复了旧代码先取消 scope 导致释放任务永远不执行的问题。
- Honor Magic8 Pro / SM8850 真机对 OpenCL 存储模式的结论：Flow `Buffer(68)` 热态约 0.81-1.08 秒，明显快于 `自动(4)` 约 2.5 秒和 `Image(132)` 约 3.75 秒；HiFT 整图 OpenCL 约 2.89 秒，比 CPU 慢。原因是 HiFT 动态长度、小算子和数据重排/搬运开销大，CPU 路径中约 90% 时间实际集中在 HiFT core。
- `cosytest` 默认改为 Flow Buffer(68) + HiFT CPU；LLM 运行时配置增加 MNN `power=high`，不修改或导出原始模型配置。强制冷启动后的两次完整真机合成分别为 4.60 秒音频/4.51 秒耗时（RTF 0.98）和 5.20 秒音频/5.00 秒耗时（RTF 0.96）；第二次 LLM 1.59 秒、Flow 0.81 秒、HiFT 2.42 秒。
- 测试页新增“导出 ZIP”，使用 SAF 输出根目录扁平、无压缩的 17 文件完整模型包，可由同页“导入 ZIP”恢复。真机导出位于 `/sdcard/阅读/cosyvoice3-mnn-complete.zip`，1,399,294,349 bytes，SHA-256 `1DA42B52E120EADAE22A29BA4E5F8B6AA236329624D7A6339538B745E7A56757`；`unzip -t` 全部通过，中央目录确认 17 文件、解压总大小 1,399,078,584 bytes。
- 当前安装包为 `3.26.07201229-cosytestdebug`；APK 位于 `E:\AndroidStudioProjects\legado_cosytest_3.26.07201229_460720122.apk`，64,435,350 bytes，SHA-256 `7490818A038821A1A7889964B1B5D4196E3617C3511FCB47A96BB92AE2974304`。
- 本机没有 QAIRT/QNN SDK，MNN QNN backend 需要 `QNN_SDK_ROOT`，因此尚未做真实 HTP/NPU 测试。下一步架构优化优先合并三个 JNI 库和 Conditioner 为单一 native runtime，用内存张量传递取代阶段文件与进程启动，并为阅读场景实现跨句 LLM/Flow/HiFT 流水线；不采用用户已拒绝的缓存水位暂停/背压。

## 2026-07-20 HiFT CPU/6 与音色注册方案

- 使用同一份 220 帧 Mel 对 HiFT CPU High/Normal/Low 和 2/4/6/8 线程做真机受控基准。High 与 Normal 在所有线程数下 WAV 逐样本一致；High/6 的 core 约 1.35-1.39 秒，优于 High/4 的约 1.66-1.71 秒，8 线程反而不稳定。Low/6 虽约 0.77-0.79 秒，但相对 High 增益约 +2.267 dB、SNR 仅 8.24 dB、最大样本误差约 0.761，禁止用于产品。
- App 首次只把 `threads=6` 传给 F0，而 CPU HiFT core 仍误用 `hiftGpuMode=4` 作为 MNN CPU 线程数，因此没有获得基准收益。`CosyVoiceNative.kt` 已修正：CPU core 与 F0 都使用 6，只有 OpenCL core 才传 4/68/132 GPU mode。
- 修正后 BKQ_AN90 真机冷启动 4.80 秒音频总耗时 8.76 秒，其中首次 Flow Shape 3.39 秒、HiFT 1.95 秒；同进程热态两次分别为 5.36 秒音频/4.76 秒（RTF 0.89，HiFT 1.96 秒）和 4.76 秒音频/4.26 秒（RTF 0.90，HiFT 1.67 秒）。
- 已安装 `3.26.07201325-cosytestdebug`，APK 为 `E:\AndroidStudioProjects\legado-archive-v3-3.26.06071119\app\build\outputs\apk\cosytest\debug\legado_cosytest_3.26.07201325_460720132.apk`，64,435,189 bytes，SHA-256 `D6091A4C4C05C9F6DF3FB622D5BFB97028A9D892164AA27265F75A372F0E6C33`。
- CosyVoice3 零样本音色必须拆成低频“音色注册”和高频“实时合成”。注册时一次性从 MP3/WAV 提取 speech tokens、24 kHz 80 维 prompt Mel、192 维 CAMPPlus embedding 和参考文本 tokens；实时句子只读取音色档案，不再解析 MP3 或运行注册模型。
- 较长干净参考音频可用于 CAMPPlus speaker embedding，但实时 prompt 应自动选取最稳定的 3-5 秒，并限制约 125 speech tokens/250 Mel frames；把整段长音频放进每句 Flow 上下文会持续拖慢实时合成。
- 官方 `speech_tokenizer_v3.onnx` 为 969,451,503 bytes，`campplus.onnx` 为 28,303,423 bytes，应作为约 1 GB 的可选音色创建扩展下载/导入，不进 APK；创建音色后可删除扩展而保留轻量档案。当前固定 `PROMPT_TOKENS=87`、`PROMPT_FRAMES=174` 必须迁移为每个 profile 的动态值。
- `instruct2` 的 LLM prompt 是 instruction text，并移除 LLM prompt speech tokens；Flow 仍使用音色档案的 prompt speech tokens、prompt feat 和 speaker embedding。语言、方言、情绪和表达风格由结构化指令控制；最终实际响度继续由 LUFS/播放器确定，不能只依赖生成指令。
- 完整实施文档：`research\mnn-cosyvoice3\VOICE_ENROLLMENT_AND_INSTRUCT_PLAN.md`。

## 2026-07-20 默认音色 profile 迁移

- 新增 `CosyVoiceVoiceProfile` 和 `files/cosyvoice3-mnn/voices/<voiceId>` 档案目录。运行时不再使用硬编码 87 prompt tokens/174 frames，而是从 profile 读取 prompt prefix、token/frame 长度及三个运行时条件文件。
- 默认 `builtin-mnn-reference-v1` 从原 17 文件模型包兼容迁移 `prompt-speech-tokens.csv`、`prompt-cond.bin`、`spks.bin`；共享 4.8 MB `rand-noise.bin` 使用应用私有软链接，不复制到每个音色。模型 ZIP 规格与用户已导入模型均不变。
- 静态校验确认三个默认音色源文件 SHA-256 与模型规格一致，大小为 400/55,680/320 bytes，87 tokens 与 174 frames 对齐；默认 profile hash 为 `4d7860df818662004aca78e0696d2f43b3de49d14a1623ada249cd8d4ea3daea`。
- `compileCosytestDebugKotlin`、`testCosytestDebugUnitTest` 和 `assembleCosytestDebug` 均通过。APK `E:\AndroidStudioProjects\legado_cosytest_3.26.07201340_460720134.apk` 为 64,442,279 bytes，SHA-256 `A6EBB55A69BD0F90875156118A70DFE6D6F7EA89C18B9A727B5D28BA7510F2E2`，已通过 `pm install -r -d` 覆盖安装。
- USB 恢复后完成真机验收：`profile.json` 正确生成，profile hash 为 `4d7860df818662004aca78e0696d2f43b3de49d14a1623ada249cd8d4ea3daea`；三个运行时文件与 model 源文件 SHA-256 逐一相同，`rand-noise.bin` 正确软链接到共享模型文件。
- profile 迁移后的冷启动 4.64 秒音频/8.92 秒（含 Flow 首次 Shape 3.56 秒），热态两次为 4.32 秒音频/4.06 秒（RTF 0.94）和 4.32 秒音频/3.42 秒（RTF 0.79）。PCM peak/RMS 正常，结果页显示 `音色 MNN 基准音色`；profile mtime 在两次热态合成前后保持 `1784527091`，证明没有按句重建。

## 2026-07-20 CosyVoice3 Instruct2 真机分支

- MNN LLM JNI 增加 `appendPromptSpeechTokens` 参数。`ZERO_SHOT` 保持追加音色 profile 的 speech tokens；`INSTRUCT2` 只输入规范化 instruction prompt，不向 LLM 附加参考 speech tokens，Flow 仍使用同一音色 profile。
- `cosytest` 测试页新增“普通复刻/指令演绎”、11 个官方来源预设和自由指令输入；用户输入中的换行和 `<|endofprompt|>` 会被清理，只允许运行时追加一个结束标记。
- BKQ_AN90 真机普通复刻报告为 `promptSpeechTokensAppended=true`，127 tokens、5.08 秒 WAV；四川话为 `false`，129 tokens、5.16 秒 WAV、热态 RTF 0.87；快速指令为 `false`，77 tokens、3.08 秒 WAV、热态 RTF 0.80。三组 token/WAV 哈希均不同，PCM 指标正常。
- 当前只确认指令分支和速度控制真实生效；方言、情绪和风格的主观质量仍需人工听测与批量矩阵验证。测试 APK 为 `E:\AndroidStudioProjects\legado_cosytest_3.26.07201412_460720141.apk`，64,452,952 bytes，SHA-256 `AF120525CD5C61CE838915B12A857E259A0C15CC1451A64F770F90D4251E9081`，已覆盖安装且保留模型数据。

## 2026-07-20 多角色 AI 表达参数

- 多角色 AI 在同一次角色分配请求中为每个 character/thought unit 返回 `languageCode/accentCode/emotionCode`，不增加逐句 AI 请求。默认固定为 `zh/mandarin/neutral`，只有正文有明确外语、方言或情绪证据时才覆盖。
- 语言目录为 9 种：中文、英文、日文、韩文、德文、西班牙文、法文、意大利文、俄文；中文方言/口音目录为 18 种：普通话、广东话、闽南话、四川话、东北话、陕西话、上海话、天津话、山东话、宁夏话、甘肃话、贵州话、河南话、湖北话、湖南话、江西话、山西话、云南话；非中文使用 `standard`。
- 结构化情绪为中性、开心、伤心、生气、害怕、惊讶、焦虑、激动、温柔、严肃；与 HTTP TTS 的 `emotionName/emotionTag` 保持分离。
- 参数已进入角色缓存、预览明细、语音计划、路由签名和音频缓存键。角色缓存 schema 为 v22，预处理版本为 v16，段落 JSON schema 为 5；旧缓存缺字段时自动迁移到默认值。
- 已知角色台词也进入同一 AI 批次以取得表达参数；AI 返回空角色名时保留本地已确认角色，不能把角色清空。正式 `app` 仍使用现有 Edge/HTTP/系统 TTS，是否支持指定语言或方言取决于具体引擎；`cosytest` 已能把结构化参数转换为 Instruct2。
- `ReadAloudPerformanceTest` 与 `CosyVoiceInstructionTest` 均通过；Cosy 测试移到 `src/testCosytestDebug`，避免阻断正式 app 单测。正式 Debug APK 为 `E:\AndroidStudioProjects\legado_app_3.26.07201513_460720151.apk`，53,283,102 bytes（50.81 MiB），SHA-256 `CBA3235DF670529530310C0070A5817B5FD23F8C7A5C4FB82BEBB2929D9FBE3C`，包内没有 Cosy/MNN/OpenCL 文件，未安装到手机。

## 2026-07-21 本地/HTTPS 混合朗读队列

- “角色分配完成后读两句便转圈”的根因是前台只把最初 2 个 `MediaItem` 加入 ExoPlayer，后续请求仅调用缓存准备而未入队。角色计划已就绪时现在一次创建当前剩余整章请求；计划未就绪的后续临时请求也改为合成后按 cue 顺序入队。
- 合成调度拆为三个独立通道：HTTPS 使用用户配置的并发数，本地 CosyVoice 固定 1，系统 TTS 固定 1。各通道可同时合成，互不占用并发名额；最终缓存身份和播放器队列仍统一按正文 cue 顺序处理。
- `LocalVoiceProgressProvider` 把 CosyVoice 的 LLM、Flow 输入、Flow GPU、HiFT 和完成阶段送入 `ReadAloudSynthesisState`；沉浸页显示通道、音色、阶段和近似阶段百分比，总进度包含活动阶段。
- 自动配音时，主角/重要角色会优先选择明确音色偏好或性别、年龄分类匹配的本地音色；普通角色仍优先 HTTPS，旁白默认仍由云健策略决定。CosyVoice 同步不再覆盖用户手动修改的音色分类，本地音色分类也不再伪装成 HTTP 发言人组。
- Honor BKQ_AN90 真机验证：当前章剩余 23 段进入同一任务，进度从 0 推进到 7；诊断日志显示初始 2 条后连续追加到 `total=17`，播放器实际跨过 cue 36、37、38，转场日志为 `playlist=2/17`、`3/17`，证明不再在第二句结束队列。
- `compileCosytestDebugKotlin`、`testCosytestDebugUnitTest`、`compileAppDebugKotlin`、`assembleCosytestDebug` 均通过。已保留数据覆盖安装 `legado_cosytest_3.26.07211004_460721100.apk`，68,134,530 bytes，SHA-256 `4C8B3B021EC0AD9BB59FFDBEA37C6211C3A1B7B2B0F19053ADC68DBD5C5E6160`。
