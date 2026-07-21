# Stage 3 feasibility gate

Date: 2026-07-15

## Decision

The mobile distillation route is now supported by an engineering proof of concept: a single-branch CFG
student and a two-step macro-trajectory student both preserve the teacher mel on unseen sentences. This is
not a production release gate. Do not ship the model before broader text, speaker, listening, and Android
end-to-end validation.

The Reading Archive app remains unchanged. All weights and experiments are isolated from the APK.

## Verified environment

- Teacher: official `Fun-CosyVoice3-0.5B-2512` with `llm.rl.pt` hard-linked as `llm.pt`.
- Model revision: `29e01c4e8d000f4bcd70751be16fa94bf3d85a18`.
- GPU: NVIDIA GeForce RTX 4060 Ti 16 GB.
- Runtime: Python 3.9.19, PyTorch 2.4.1, CUDA 11.8.
- Peak allocated GPU memory: about 4.5 GB; peak reserved: about 5.6 GB.

## Measured results

Normal Chinese sentence, 6.08 seconds output:

| Flow steps | Total time | Total RTF | RMS | RMS change vs 10 | Speaker cosine vs 10 | Whisper CER |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 6.02 s | 0.990 | 0.0891 | 0.00 dB | 1.000 | 0.000 |
| 4 | 5.93 s | 0.976 | 0.1176 | +2.41 dB | 0.976 | 0.000 |
| 2 | 7.28 s | 1.198 | 0.1652 | +5.36 dB | 0.916 | 0.000 |

Tongue twister, 9.36 seconds output:

| Flow steps | Total time | Total RTF | RMS | RMS change vs 10 | Speaker cosine vs 10 |
|---:|---:|---:|---:|---:|---:|
| 10 | 11.29 s | 1.206 | 0.1190 | 0.00 dB | 1.000 |
| 4 | 9.16 s | 0.978 | 0.1486 | +1.93 dB | 0.990 |
| 2 | 8.37 s | 0.894 | 0.2129 | +5.05 dB | 0.930 |

The tongue-twister Whisper result is rejected as a content metric because the Base ASR model hallucinated
repeated homophones even for the 10-step reference. The normal sentence produced identical recognized text
for all three variants after Traditional-to-Simplified normalization.

## Interpretation

1. Four Euler steps preserve content in the tested normal sentence and keep the speaker embedding close,
   so a real 10-to-4-step distillation experiment is justified.
2. Merely reducing the solver to four steps already changes loudness by about 2 dB and changes the acoustic
   spectrum. It is not a drop-in production optimization.
3. Two steps cause a consistent roughly 5 dB loudness increase and a much larger speaker drift. The raw
   two-step route is rejected.
4. Desktop end-to-end speed is dominated by LLM, frontend, and fixed costs for short sentences. Lower Flow
   step count does not guarantee lower total RTF; one normal-sentence two-step run was slower than ten steps.
5. On the phone, Flow remains a larger bottleneck, but the existing MNN OpenCL model is only numerically
   valid in High precision. Distillation does not automatically solve the earlier FP16 attention overflow.

## Next gate

Expand the two-step student to at least 100 Chinese audiobook sentences and multiple unseen speakers. Use
blind listening, ASR CER, CAMPPlus similarity, loudness delta, clipping, and an actual Android end-to-end
RTF test. The 24-sentence, one-speaker proof of concept is too small for release even though its held-out mel
metrics pass the engineering gate.

Do not begin the 0.3B LLM student until the two-step Flow student passes the broader gate. A 4060 Ti can run
the offline cache and selective fine-tuning workflow, but a production student requires a licensed corpus.

## Android heterogeneous benchmark

Measured on Honor BKQ_AN90 / SM8850 / Adreno with MNN 3.6.0:

| Module | Backend | Precision | Measured result | Decision |
|---|---|---|---:|---|
| CosyVoice LLM INT4 | CPU, 4 threads | Low | 84.99 token/s median with RAS | Keep on CPU |
| CosyVoice LLM INT4 | OpenCL, warm | Low | 68.38 token/s median with RAS | Reject for production |
| Flow estimator, seq 414 | OpenCL | High | 1.838 s per estimator call | GPU path is correct but too slow |
| HiFT core, 174 frames | CPU | High | 0.934 s | Keep on CPU |
| HiFT core, 174 frames | OpenCL | High | 1.315 s | Reject for production |

The phone LLM test used 87 prompt-speech tokens and the real
`prompt_text + target_text + prompt_speech_tokens` layout. A CosyVoice-specific RAS sampler generated
118-125 target speech tokens plus EOS in three CPU runs, with no out-of-range token. OpenCL also generated
valid speech tokens and EOS, but its warm decode was slower. Its cold prefill took about 10.09 seconds while
compiling and caching kernels, so it is unsuitable for first-sentence latency.

For the median 120 target speech tokens, the target duration is approximately 4.8 seconds at the model's
25 Hz token rate. The Flow estimator operates on about
`2 * (87 prompt + 120 target) = 414` mel positions. At four estimator calls, Flow alone takes about
7.35 seconds, or RTF 1.53 before the LLM, HiFT, native DSP, and frontend. The current full-size model therefore
fails the phone RTF < 1 gate even though GPU execution is numerically correct.

The current best runtime split is CPU Low/4 threads for LLM, OpenCL High for Flow, and CPU High for HiFT.
The next performance experiment must reduce Flow compute structurally. Backend switching, more synthesizer
workers, and raw solver-step reduction cannot close the measured gap.

### CFG single-branch gate

The official Flow estimator export fixes classifier-free guidance to batch 2. A separate fixed-batch-1
ONNX/MNN graph was exported from the same weights and tested at sequence 414. Its external weight file is
byte-identical to the batch-2 graph, so only the small graph differs.

| Flow graph | OpenCL High median | FLOPs | ONNX max abs error |
|---|---:|---:|---:|
| Batch 2 CFG teacher | 1.838 s | 172625 M | 0.00326 |
| Batch 1 conditional branch | 0.912 s | 86304 M | 0.00373 |

Batch 1 is 49.6% of the batch-2 latency. This is a valid performance gate, not a production model: removing
the unconditional branch changes the guided velocity and therefore the generated audio. The recommended
student is a CFG-distilled single-branch estimator trained to predict the teacher's guided velocity, followed
by two-step trajectory/consistency distillation. At the measured speed, two single-branch Flow calls cost
about 1.82 seconds. Combined with streaming overlap between LLM, Flow, and HiFT, this route can plausibly pass
steady RTF < 1; sequential one-shot latency will remain near the threshold and needs separate optimization.

### CFG and two-step distillation proof of concept

The teacher cache records the exact ten-step solver inputs and the guided target
`1.7 * v_cond - 0.7 * v_uncond`. Twenty-four real Chinese synthesis trajectories were captured with one
zero-shot prompt: 20 utterances for training and 4 whole utterances for validation. The last 8 of 22 DiT
blocks plus the output layers were trained, or 119,728,208 of 331,142,224 estimator parameters. Peak CUDA
allocation was 3.27 GB on the RTX 4060 Ti.

| Variant on an unseen sentence | Steps | Mel cosine vs teacher | Mel SNR vs teacher |
|---|---:|---:|---:|
| CFG student | 10 | 0.99920 | 27.93 dB |
| CFG student, raw step reduction | 4 | 0.99489 | 19.82 dB |
| CFG student, raw step reduction | 2 | 0.98172 | 14.13 dB |
| Two-step macro-trajectory student | 2 | 0.99843 | 24.99 dB |

Across the four held-out cache utterances, the two-step student's generated-only mel region averaged
0.99882 cosine and 26.26 dB SNR against the ten-step CFG teacher. Waveform-level correlation is not used as
the primary gate because HiFT phase is highly sensitive to small mel changes; finite output, clipping,
loudness, listening, CER, and speaker similarity remain required production metrics.

The two-step target uses cosine-schedule time points `0 -> 0.2928932188 -> 1`. Each macro target is the
average velocity needed to move between the corresponding dense teacher trajectory states. Directly running
the local-velocity student for two steps is rejected; macro-trajectory training is required.

Final Android MNN results on Honor BKQ_AN90 / SM8850 / Adreno:

| Model | Shape | Backend | Median per call | Two-call Flow | MNN max abs vs ONNX |
|---|---|---|---:|---:|---:|
| Two-step batch-1 student | `[1,80,516]` | OpenCL High | 1.187 s | 2.374 s | 0.00551 |

OpenCL Normal and Low both measured about 0.633 seconds per call but returned non-finite output, so they are
rejected. High precision is mandatory. A text-to-WAV Android runner has now been completed, but the production
gate has not passed because the current stages execute sequentially and repeated runs expose LLM throughput
degradation.

## Android text-to-WAV result

The phone pipeline now runs MNN LLM on CPU Low/4 threads, a dynamic token-to-Flow conditioner, two-step Flow
on OpenCL High, HiFT on CPU High/4 threads, and native harmonic source/STFT/iSTFT code.

The fixed 516-frame numerical case passed with a two-step Flow median of 2.405 s and final mel maximum error
0.01099. The 342-frame HiFT path took 2.711 s; F0, source STFT, and PCM maximum errors were 0.00229, 0.00174,
and 0.01614. The reconstructed 6.84 s WAV was finite and unclipped.

A live text run generated 124 speech tokens, 4.96 s of audio, and achieved Whisper Base CER 0 with peak
0.574, RMS 0.0644, and no clipping. A later one-command run generated 139 tokens and 5.56 s audio, but the
LLM slowed from about 85 token/s to 42.6 token/s after repeated benchmarks. Sequential time was 8.18 s and
RTF 1.47; Whisper CER was 4.76% with no clipping. This variability blocks APK integration.

MNN external Flow inputs must be restored before both estimator calls. Reusing static `mask/mu/spks/cond`
tensors changed the final mel by up to 12.1 because the optimized graph aliases input storage. HiFT remains
at 4 threads because 6 and 8 threads were slower under sustained load.

Persistent-session module tests are now complete. LLM stayed near 83-85 token/s over repeated isolated
requests, Flow reused fixed shape buckets, and HiFT reused both CPU sessions. A corrected cold one-shot run
generated 3.92 seconds of audio in 20.04 seconds wall time (cold RTF 5.11); 12.73 seconds came from Flow load
and first OpenCL shape compilation. This confirms that app startup must prewarm the common Flow bucket.

With Flow bucket 512 prewarmed, a synchronized three-process test ran LLM on CPU, Flow on Adreno OpenCL High,
and HiFT on CPU at the same time. A cool-device run completed 14.8 seconds of work in 9.33 seconds, for peak
throughput RTF 0.63. All workers returned zero and PCM peaks were 0.58-0.82. A second self-contained run,
started immediately after 58 seconds of preparation load, completed 15.12 seconds of work in 19.51 seconds,
RTF 1.29. Flow remained near 7.94 seconds but LLM and HiFT CPU work roughly doubled. Thermal status was 1 and
skin temperature was near the light-throttling threshold. Sustained real-time performance therefore remains
unproven; the product scheduler needs cache-ahead backpressure rather than continuous maximum load.

The benchmark now supports separate OpenCL modes for Flow and HiFT core and persistent MNN kernel/tuning
caches. Mode 4 (Wide/automatic memory), 68 (Wide/Buffer), and 132 (Wide/Image) must be compared after device
reconnection. Moving HiFT core to GPU is a system-level candidate because it may reduce CPU contention even
if isolated HiFT GPU latency is slower; the final decision must use whole-pipeline thermal throughput.

The FP16 Flow candidate reduces the complete external model set to 1,394,278,088 bytes (1.299 GiB). It keeps
phone speed roughly unchanged and passed the current numerical/single-sentence ASR check. Flow INT8 is
rejected because final-mel error rose to about 1.95 without a latency benefit.

The next implementation step is a single persistent JNI runtime with a dependency-driven sentence queue,
explicit cancellation, fixed-bucket prewarming, and atomic output. Only the independent `cosytest` flavor may
host it initially. The old CrispASR/Vulkan backend must be removed from that flavor, and arbitrary user-audio
cloning must stay disabled because the current mobile assets contain only one preprocessed reference voice.
Formal Reading Archive integration still requires repeated warm RTF < 1 plus the 100-sentence, multi-speaker
quality gate.
