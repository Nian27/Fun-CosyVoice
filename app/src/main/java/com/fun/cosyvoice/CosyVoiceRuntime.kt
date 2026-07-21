package com.fun.cosyvoice

import android.util.Log
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.job
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicReference

data class CosyVoiceSynthesisOptions(
    val flowGpuMode: Int = 68,
    val hiftCoreBackend: String = "cpu",
    val hiftGpuMode: Int = 4,
    val voiceProfileId: String = CosyVoiceStore.DEFAULT_VOICE_PROFILE_ID,
    val inferenceMode: CosyVoiceInferenceMode = CosyVoiceInferenceMode.ZERO_SHOT,
    val instruction: String = ""
) {
    init {
        require(flowGpuMode in setOf(4, 68, 132))
        require(hiftCoreBackend in setOf("cpu", "opencl"))
        require(hiftGpuMode in setOf(4, 68, 132))
        if (inferenceMode == CosyVoiceInferenceMode.INSTRUCT2) {
            require(instruction.isNotBlank())
        }
    }
}

data class CosyVoiceSynthesisReport(
    val output: File,
    val totalMs: Long,
    val llmMs: Long,
    val conditionerMs: Long,
    val flowResizeMs: Double,
    val flowInferenceMs: Double,
    val hiftMs: Double,
    val audioSeconds: Double,
    val targetTokens: Int,
    val flowBucket: Int,
    val pcmPeak: Double,
    val pcmRms: Double,
    val voiceProfile: CosyVoiceVoiceProfile,
    val options: CosyVoiceSynthesisOptions
) {
    val rtf: Double
        get() = totalMs / 1000.0 / audioSeconds.coerceAtLeast(0.001)

    fun displayText(): String = buildString {
        append("合成完成 · %.2f 秒音频 · 总耗时 %.2f 秒 · RTF %.2f".format(audioSeconds, totalMs / 1000.0, rtf))
        append("\nLLM %.2f 秒 · Flow %.2f 秒（首次/形状 %.2f 秒）· HiFT %.2f 秒".format(
            llmMs / 1000.0, flowInferenceMs / 1000.0, flowResizeMs / 1000.0, hiftMs / 1000.0
        ))
        append("\n音色 ${voiceProfile.displayName} · ${options.inferenceMode.displayName()} · Flow GPU mode ${options.flowGpuMode} · HiFT ${options.hiftCoreBackend} · peak %.3f · rms %.3f".format(pcmPeak, pcmRms))
    }
}

internal object CosyVoiceRuntime {
    private const val TAG = "CosyVoiceRuntime"
    private const val HIFT_CPU_THREADS = 6
    private val mutex = Mutex()
    private val activeProcess = AtomicReference<Process?>(null)

    suspend fun synthesize(
        store: CosyVoiceStore,
        text: String,
        output: File,
        options: CosyVoiceSynthesisOptions,
        onStage: suspend (String) -> Unit = {}
    ): CosyVoiceSynthesisReport = mutex.withLock {
        check(store.modelStatus().ready) { "MNN 模型不完整" }
        val cleanText = text.replace(Regex("[\\r\\n]+"), " ").trim()
        require(cleanText.isNotEmpty()) { "试听文字不能为空" }
        val voiceProfile = store.voiceProfile(options.voiceProfileId)
        check(voiceProfile.promptTokenCount <= CosyVoiceStore.MAX_REALTIME_PROMPT_TOKENS) {
            "音色提示过长（${voiceProfile.promptTokenCount} Token），请先点\"实时版\"生成朗读音色"
        }

        val startedAt = System.nanoTime()
        val runDir = File(store.workDir, "run-${System.currentTimeMillis()}").apply {
            deleteRecursively()
            mkdirs()
        }
        val llmOutput = File(runDir, "llm-output").apply { mkdirs() }
        val flowInput = File(runDir, "flow-input").apply { mkdirs() }
        val flowOutput = File(runDir, "flow-output").apply { mkdirs() }
        val hiftInput = File(runDir, "hift-input").apply { mkdirs() }
        val hiftOutput = File(runDir, "hift-output").apply { mkdirs() }
        val model = store.modelDir

        val promptFile = File(runDir, "prompts.txt").apply {
            val prefix = if (options.inferenceMode == CosyVoiceInferenceMode.INSTRUCT2) {
                CosyVoiceInstruction.promptPrefix(options.instruction)
            } else {
                voiceProfile.promptPrefix
            }
            writeText(prefix + cleanText, Charsets.UTF_8)
        }

        onStage("LLM 正在生成语音 Token")
        val llmStartedAt = System.nanoTime()
        val llmExitCode = CosyVoiceLlmNative.run(
            configPath = store.llmRuntimeConfig().absolutePath,
            promptsPath = promptFile.absolutePath,
            maxTokens = 500,
            promptSpeechTokensPath = voiceProfile.promptSpeechTokensFile.absolutePath,
            outputDirectory = llmOutput.absolutePath,
            appendPromptSpeechTokens = options.inferenceMode == CosyVoiceInferenceMode.ZERO_SHOT
        )
        val llmStage = (System.nanoTime() - llmStartedAt) / 1_000_000L
        check(llmExitCode == 0) { "LLM JNI 执行失败($llmExitCode)" }
        currentCoroutineContext().ensureActive()
        val tokenFile = File(llmOutput, "speech-tokens-0.csv")
        val targetTokens = tokenFile.readText().split(',').count { it.trim().isNotEmpty() }
        check(targetTokens > 0) { "LLM 没有生成语音 Token" }
        val sequenceLength = 2 * (voiceProfile.promptTokenCount + targetTokens)
        val targetFrames = 2 * targetTokens
        val flowBucket = flowBucket(sequenceLength)

        onStage("正在生成 Flow 输入")
        val conditionerStage = runCommand(
            stage = "conditioner",
            command = listOf(
                store.executable("libcosy_conditioner_exec.so").absolutePath,
                File(model, "flow-conditioner.fp32.mnn").absolutePath,
                voiceProfile.directory.absolutePath,
                tokenFile.absolutePath,
                flowInput.absolutePath,
                voiceProfile.promptTokenCount.toString(),
                voiceProfile.promptFrameCount.toString()
            ),
            directory = runDir,
            logFile = File(runDir, "conditioner.log")
        )
        currentCoroutineContext().ensureActive()

        onStage("Flow 正在使用 GPU 合成 Mel")
        val flowManifest = File(runDir, "flow-manifest.txt").apply {
            writeText(
                "${flowInput.absolutePath} ${flowOutput.absolutePath} $sequenceLength ${voiceProfile.promptFrameCount} $flowBucket\n",
                Charsets.US_ASCII
            )
        }
        val flowReportFile = File(runDir, "flow-report.jsonl")
        val flowCache = File(store.gpuCacheDir, "flow-fp16-opencl-high-mode${options.flowGpuMode}.cache")
        val flowExitCode = CosyVoiceFlowNative.run(
            modelPath = File(model, "flow.cfg-student-2step.batch1.fp16.mnn").absolutePath,
            manifestPath = flowManifest.absolutePath,
            backend = "opencl",
            precision = "high",
            threads = options.flowGpuMode,
            reportPath = flowReportFile.absolutePath,
            cachePath = flowCache.absolutePath
        )
        check(flowExitCode == 0) { "Flow JNI 执行失败($flowExitCode)" }
        currentCoroutineContext().ensureActive()

        File(flowOutput, "student_target_mel_android.bin").copyTo(
            File(hiftInput, "mel.bin"), overwrite = true
        )
        listOf("source-linear-weight.bin", "source-linear-bias.bin").forEach { name ->
            File(model, name).copyTo(File(hiftInput, name), overwrite = true)
        }
        val hiftManifest = File(runDir, "hift-manifest.txt").apply {
            writeText("${hiftInput.absolutePath} ${hiftOutput.absolutePath} $targetFrames\n", Charsets.US_ASCII)
        }
        val hiftReportFile = File(runDir, "hift-report.jsonl")
        val hiftCache = File(store.gpuCacheDir, "hift-core-${options.hiftCoreBackend}-high-mode${options.hiftGpuMode}.cache")
        onStage(
            if (options.hiftCoreBackend == "opencl") "HiFT core 正在使用 GPU 生成音频"
            else "HiFT 正在使用 CPU 生成音频"
        )
        val hiftExitCode = CosyVoiceHiFTNative.run(
            f0ModelPath = File(model, "hift-f0.fp32.mnn").absolutePath,
            coreModelPath = File(model, "hift-core.fp32.mnn").absolutePath,
            manifestPath = hiftManifest.absolutePath,
            threads = HIFT_CPU_THREADS,
            reportPath = hiftReportFile.absolutePath,
            coreBackend = options.hiftCoreBackend,
            corePrecision = "high",
            coreCachePath = hiftCache.absolutePath,
            coreGpuMode = if (options.hiftCoreBackend == "cpu") HIFT_CPU_THREADS else options.hiftGpuMode
        )
        check(hiftExitCode == 0) {
            val health = runCatching { hiftReportFile.firstJson() }.getOrNull()
            if (health == null) "HiFT JNI 执行失败($hiftExitCode)"
            else "HiFT JNI 执行失败($hiftExitCode)：finite=${health.optBoolean("pcmFinite")} peak=${
                "%.3f".format(health.optDouble("pcmPeak"))
            } rms=${"%.3f".format(health.optDouble("pcmRms"))} write=${health.optBoolean("waveWritten")}"
        }
        currentCoroutineContext().ensureActive()

        val generated = File(hiftOutput, "hift-android.wav")
        check(generated.length() > 44L) { "HiFT 没有生成有效 WAV" }
        output.parentFile?.mkdirs()
        val tempOutput = File(output.parentFile, "${output.name}.writing")
        generated.copyTo(tempOutput, overwrite = true)
        output.delete()
        check(tempOutput.renameTo(output)) { "试听音频无法保存" }

        val llm = File(llmOutput, "llm-persistent.jsonl").firstJson()
        val flow = flowReportFile.firstJson()
        val hift = hiftReportFile.firstJson()
        val totalMs = (System.nanoTime() - startedAt) / 1_000_000L
        val report = CosyVoiceSynthesisReport(
            output = output, totalMs = totalMs,
            llmMs = llmStage, conditionerMs = conditionerStage,
            flowResizeMs = flow.optDouble("resizeMs", 0.0),
            flowInferenceMs = flow.optDouble("inferenceMs", 0.0),
            hiftMs = hift.optDouble("requestMs", 0.0),
            audioSeconds = hift.optDouble("audioSeconds", targetFrames * 0.02),
            targetTokens = targetTokens, flowBucket = flowBucket,
            pcmPeak = hift.optDouble("pcmPeak", 0.0),
            pcmRms = hift.optDouble("pcmRms", 0.0),
            voiceProfile = voiceProfile, options = options
        )
        Log.d(TAG, report.displayText().replace('\n', ' '))
        onStage("合成完成，正在播放")
        report
    }

    suspend fun close() = mutex.withLock {
        activeProcess.getAndSet(null)?.destroyForcibly()
        CosyVoiceLlmNative.reset()
        CosyVoiceFlowNative.reset()
        CosyVoiceHiFTNative.reset()
    }

    private suspend fun runCommand(
        stage: String, command: List<String>, directory: File, logFile: File
    ): Long {
        currentCoroutineContext().ensureActive()
        val startedAt = System.nanoTime()
        val process = ProcessBuilder(command)
            .directory(directory)
            .redirectErrorStream(true)
            .start()
        activeProcess.set(process)
        val cancellation = currentCoroutineContext().job.invokeOnCompletion {
            if (process.isAlive) process.destroyForcibly()
        }
        val console = try {
            process.inputStream.bufferedReader().use { it.readText() }
        } finally {
            cancellation.dispose()
        }
        val exitCode = process.waitFor()
        activeProcess.compareAndSet(process, null)
        logFile.writeText(console, Charsets.UTF_8)
        val elapsedMs = (System.nanoTime() - startedAt) / 1_000_000L
        Log.d(TAG, "stage=$stage exit=$exitCode elapsed=${elapsedMs}ms log=${logFile.absolutePath}")
        check(exitCode == 0) { "$stage 执行失败($exitCode)：${console.takeLast(1200)}" }
        return elapsedMs
    }

    private fun flowBucket(sequenceLength: Int): Int =
        listOf(512, 768, 1024, 1280, 1536, 2048).firstOrNull { sequenceLength <= it }
            ?: error("文本过长：Flow sequence=$sequenceLength")

    private fun File.firstJson(): JSONObject {
        check(isFile) { "缺少报告文件：$absolutePath" }
        val line = useLines { lines -> lines.firstOrNull { it.isNotBlank() } }
            ?: error("报告为空：$absolutePath")
        return JSONObject(line)
    }
}

private fun CosyVoiceInferenceMode.displayName(): String = when (this) {
    CosyVoiceInferenceMode.ZERO_SHOT -> "普通复刻"
    CosyVoiceInferenceMode.INSTRUCT2 -> "指令演绎"
}
