# CosyVoice3 MNN Android benchmark

This directory is an isolated feasibility benchmark. It does not modify or package the Reading Archive app.

## Stage 3: RL teacher and Flow step gate

The official RL teacher now runs end to end on the desktop 4060 Ti. The same seed and text were generated
with 10, 4, and 2 Euler steps. See `STAGE3_FEASIBILITY.md` and `results/stage3-flow-steps*`.

The result supports a 4-step Flow distillation proof of concept, not a raw solver-step production change.
Raw 2-step output is rejected because speaker drift and loudness change are too large. The Reading Archive
app remains untouched.

Phone module tests now confirm the best heterogeneous placement:

- LLM: CPU Low, 4 threads, about 84.99 token/s with CosyVoice RAS.
- Flow: OpenCL High. A realistic sequence length of 414 takes about 1.838 s per estimator call.
- HiFT: CPU High. The 174-frame core takes about 0.934 s, faster than OpenCL's 1.315 s.

The Flow-only four-call estimate is about 7.35 s for roughly 4.8 s of target speech, so the current model
still fails the phone real-time gate. See `results/stage3-llm-android` and
`results/android-seq414-realistic`.

A fixed-batch-1 estimator exported from the same weights measured 0.912 s per call at sequence 414, versus
1.838 s for the batch-2 CFG graph. Its output is numerically valid but is only the conditional branch, not a
drop-in replacement. This validates the next training target: distill the teacher's guided velocity into a
single-branch student, then distill that student to two solver steps. See `models/flow-batch1` and
`results/android-seq414-batch1`.

The follow-up distillation PoC is complete. A single-branch student was trained against the teacher CFG
velocity, then continued on two macro-step trajectory targets. With 20 training and 4 held-out Chinese
utterances from one zero-shot speaker, the final two-step student reached 0.99882 mean mel cosine and
26.26 dB mean mel SNR on the held-out generated regions. An additional unseen sentence reached 0.99843 and
24.99 dB. This is an engineering gate, not enough data for release.

On the Honor device, the final batch-1 MNN estimator at sequence 516 takes 1.187 s per OpenCL High call;
two Flow calls take about 2.374 s. MNN matches ONNX with max absolute error 0.00551. OpenCL Normal and Low
produce non-finite output and remain forbidden. See `results/cfg-distill-real`,
`results/android-cfg-student-2step`, and `models/flow-cfg-student-2step`.

Core reproducible stages:

```powershell
./run-stage3-cfg-distill.ps1

# Equivalent individual commands:
$env:PYTHONPATH = 'E:\AndroidStudioProjects\cosyvoice3-distill-lab\python-deps'
$Python = 'D:\anaconda3\envs\pytorch\python.exe'
$Repo = 'E:\AndroidStudioProjects\CosyVoice-main'
$Model = 'E:\AndroidStudioProjects\cosyvoice3-distill-lab\Fun-CosyVoice3-0.5B-2512-RL'

& $Python scripts\capture_cfg_trajectory.py --repo $Repo --model $Model `
  --prompt-wav "$Repo\asset\zero_shot_prompt.wav" `
  --text-file data\cfg_distill_texts_zh.txt `
  --output results\cfg-distill-real\teacher-real-24utt.pt --validation-utterances 4
& $Python scripts\train_cfg_student.py --repo $Repo --model $Model `
  --cache results\cfg-distill-real\teacher-real-24utt.pt `
  --output results\cfg-distill-real\student-last8-real-24utt.pt `
  --steps 1000 --last-blocks 8 --learning-rate 0.00001
& $Python scripts\build_two_step_cache.py `
  --input results\cfg-distill-real\teacher-real-24utt.pt `
  --output results\cfg-distill-real\teacher-real-24utt-2step.pt
& $Python scripts\train_cfg_student.py --repo $Repo --model $Model `
  --cache results\cfg-distill-real\teacher-real-24utt-2step.pt `
  --initial-patch results\cfg-distill-real\student-last8-real-24utt.pt `
  --output results\cfg-distill-real\student-last8-real-24utt-2step.pt `
  --steps 2000 --last-blocks 8 --learning-rate 0.000005
```

```powershell
.\setup-stage3-cuda.ps1
.\download-stage3.ps1
.\run-stage3-flow-steps.ps1
```

## Stage 1 result

1. Download the official CosyVoice3 Flow ONNX and MNN 3.5.0 tools.
2. Convert Flow ONNX to MNN FP32 without weight quantization.
3. Compare MNN CPU output with ONNX Runtime on fixed inputs.
4. Run the same MNN model on Android OpenCL in High precision.
5. Record correctness, first-run time, steady-state time, peak memory, and thermal behavior.

Flow passes only when outputs contain no NaN/Inf, numerical error stays inside the declared tolerance, and repeated Android OpenCL runs complete without driver loss.

The Android model must use MNN external weights. An embedded 1.33 GB model requires another contiguous
1.33 GB allocation during `createFromFile()` and is not viable on the device. The external model keeps a
small graph file beside `flow.decoder.estimator.fp32.mnn.weight`.

Measured on Honor BKQ_AN90 / Adreno 840 with MNN 3.6.0 OpenCL High:

- Sequence 32: first run about 126 ms, steady median about 161 ms.
- Sequence 256: first run about 892 ms, steady median about 1.15 s.
- One sequence-256 estimator call has a measured steady median of 1.152 s. The official Euler solver calls
  the estimator 10 times, so the Flow compute estimate is 11.52 s before wrapper overhead. At 24 kHz and
  480 samples per mel frame, 256 frames cover 5.12 s of total mel context. This is RTF 2.25 against the
  whole sequence. If that sequence contains a 2.0 s prompt, the newly generated portion is about 3.12 s
  and its effective Flow RTF is about 3.69. The earlier "roughly 3 s" figure was this prompt-subtracted
  estimate, not an end-to-end generated WAV measurement.

Stage 1 is numerically correct in High precision but fails the real-time performance gate. It must not be
integrated into the Reading Archive app as-is.

The CPU oracle must restore all six external inputs before every independent MNN run. The converted
dynamic graph reuses storage that aliases input buffers; calling `runSession()` repeatedly without
rewriting inputs changes the result after the first run.

Build and run the Android ARM64/OpenCL benchmark without installing an APK:

```powershell
.\build-stage1-android.ps1
.\run-stage1-android.ps1
```

## Download

```powershell
.\download-stage1.ps1
```

The downloader keeps `.part` files and resumes HTTP Range downloads. Final files are accepted only after exact size and SHA-256 verification.

OpenCL Normal/FP16 initially overflows in the Mish activation because the converted Softplus computes
`exp(x)` directly. `scripts/stabilize_softplus_onnx.py` rewrites Softplus to the equivalent stable form
`max(x, 0) + log(1 + exp(-abs(x)))` before MNN conversion.

The tanh-GELU subgraphs also cube their inputs before saturation. In FP16 that intermediate can overflow;
`scripts/stabilize_gelu_onnx.py` clips only the cubic branch to `[-10, 10]` while preserving the original
input on the final multiplication.

Attention score MatMul outputs can exceed the FP16 finite range even when the final Softmax is well
defined. The experimental `scripts/stabilize_attention_onnx.py` rewrite changed the final result too much
and is rejected. OpenCL Normal/FP16 is not a valid production path for this model without true per-op
mixed precision.

Weight-only INT8 reduces the external weight file to about 335 MB but does not improve OpenCL latency and
increases numerical error, so it is also rejected for production.

## Stage 2: HiFT

Only the official `hift.pt` and pinned `cosyvoice3.yaml` are downloaded; the full 9.75 GB repository is
not required.

```powershell
.\download-stage2.ps1
.\setup-stage2-export-env.ps1
.\export-stage2-hift.ps1
.\convert-stage2-mnn.ps1
.\build-stage2-android.ps1
.\run-stage2-android.ps1
```

HiFT is split deliberately instead of exporting random source generation and complex STFT/iSTFT into one
opaque graph:

1. F0 predictor network -> MNN.
2. Deterministic harmonic source and STFT -> native C++ DSP.
3. HiFT convolution core -> MNN CPU High with 4 threads.
4. iSTFT overlap-add -> native C++ DSP.

## Phone end-to-end runner

The isolated Android pipeline now accepts text and runs MNN LLM -> Flow conditioner -> two-step Flow ->
HiFT F0/source/core/iSTFT without using a desktop intermediate tensor. Large model files remain under
`/data/local/tmp` and are not packaged into an APK.

```powershell
.\run-phone-e2e.ps1 -Text '今天天气很好，我们开始测试手机端语音合成速度。'
```

The script writes logs and WAV output under `results\phone-e2e\<timestamp>` and copies the latest WAV to
`/sdcard/Download/cosyvoice3-mnn-latest.wav`. This is a research runner, not the Reading Archive runtime.

This boundary makes audio corruption attributable and keeps the large convolution workload on the GPU.

## Persistent and concurrent phone results

The one-shot runner now reports both inference-only time and real cold wall time. On the Honor BKQ_AN90,
one 3.92-second sentence measured 4.97 seconds of module inference (RTF 1.27), but 20.04 seconds from process
start to WAV (cold RTF 5.11). Flow model loading and first-shape OpenCL compilation accounted for 12.73
seconds. The old inference-only total must not be used as first-play latency.

The three main modules now have persistent benchmark processes. Three consecutive sentences produced 14.8
seconds of audio with the following warm, non-overlapped work:

- LLM request work: 5.53 seconds.
- Two-step Flow inference: 7.18 seconds.
- HiFT synthesis: 5.53 seconds.
- Total warm work: 18.23 seconds, RTF 1.23.

Fixed Flow buckets are mandatory. The first unseen dynamic shape compiles for about 12-17 seconds. Padding a
real sequence of 452 to bucket 512 produced a bit-identical target mel for the real region, while subsequent
bucket-512 requests avoided recompilation. Recommended initial buckets are 512, 768, 1024, 1280, 1536, and
2048, with 512 prewarmed before playback.

An automated cross-module concurrency benchmark prewarms Flow bucket 512, then starts persistent LLM (CPU),
Flow (Adreno OpenCL High), and HiFT (CPU) at a common barrier. A cool-device run processed 14.8 seconds of
work in 9.33 seconds, throughput RTF 0.63. All workers exited successfully and PCM peaks were 0.58-0.82.
However, running the same concurrency test immediately after a full 58-second preparation load took 19.51
seconds for 15.12 seconds of audio work, RTF 1.29. Flow stayed near 7.94 seconds, while CPU LLM and HiFT
slowed to 18.93 and 14.29 seconds. Thermal status was 1 and skin temperature was near its light-throttling
threshold. The cool result proves peak cross-sentence throughput; the hot result means sustained real-time
operation is not yet proven.

```powershell
.\run-phone-persistent-batch.ps1
.\run-phone-concurrent-benchmark.ps1 -SkipPreparation
```

Omit `-SkipPreparation` for a self-contained run that regenerates the intermediate inputs first.
Use `-CooldownSeconds` to separate preparation heat from the concurrency measurement. The script records
thermal state before and after the synchronized section.

The benchmark also exposes OpenCL scheduling choices independently for Flow and the HiFT core. Mode 4 is
Wide tuning with automatic memory selection, 68 is Wide + Buffer, and 132 is Wide + Image. MNN kernel and
tuning caches are persisted outside the per-run directory, keyed by model role, precision, and mode. GPU
cache reuse and GPU HiFT scheduling are compiled but still require a connected-device comparison.

The app runtime must use cache-ahead backpressure: synthesize until a small audio buffer target is reached,
then sleep while playback consumes it. Running all stages continuously at maximum speed wastes power and
triggers CPU throttling even when peak RTF is below 1.

The current FP16 Flow candidate reduces the full external runtime bundle to 1,394,278,088 bytes (1.299 GiB).
The model must be downloaded or imported after installation and must never be packaged in the APK. Flow INT8
is rejected: it did not improve latency and produced a final-mel maximum error of about 1.95. FP16 Flow passed
the current phone numerical and single-sentence ASR check, but still needs the broader quality gate.

APK integration remains blocked until the benchmark code is turned into one persistent JNI runtime with a
dependency-driven sentence queue, cancellation, lifecycle-safe model ownership, bucket prewarming, and
atomic WAV/cache output. The existing `cosytest` CrispASR/Vulkan runtime is a failed backend and must be
replaced rather than wrapped. The current mobile voice assets represent one preprocessed reference voice;
arbitrary reference-audio cloning is not implemented and must not be exposed in the test UI yet.
