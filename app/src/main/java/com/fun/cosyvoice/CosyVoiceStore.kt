package com.fun.cosyvoice

import android.content.Context
import android.net.Uri
import android.system.Os
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

data class CosyVoiceModelStatus(
    val ready: Boolean,
    val installedBytes: Long,
    val missingFiles: List<String>
)

data class CosyVoiceModelFileSpec(
    val name: String,
    val bytes: Long,
    val sha256: String
)

class CosyVoiceStore(context: Context) {
    companion object {
        private const val TAG = "CosyVoiceStore"
        const val SAMPLE_RATE = 24_000
        const val FIXED_VOICE_NAME = "MNN 基准音色"
        const val DEFAULT_VOICE_PROFILE_ID = "builtin-mnn-reference-v1"
        const val MODEL_ID = "Fun-CosyVoice3-0.5B-2512-RL-distilled-v1"
        const val MAX_REALTIME_PROMPT_TOKENS = 125
        private const val VOICE_PROFILE_SCHEMA = 1
        private const val PREFERENCES_NAME = "cosyvoice3-mnn"
        private const val SELECTED_PROFILE_KEY = "selected_voice_profile_id"
        private const val DEFAULT_PROMPT_TOKENS = 87
        private const val DEFAULT_PROMPT_FRAMES = 174
        private const val SYSTEM_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
        private const val PROMPT_PREFIX = SYSTEM_PROMPT_PREFIX + "希望你以后能够做的比我还好呀。"

        val MODEL_FILE_SPECS = listOf(
            CosyVoiceModelFileSpec("config-cpu-cosyvoice-ras.json", 504L, "448256EFCC585A151DAB804A8C19F5CC0F6EF846DDAA294D406FF6F3FC2DC081"),
            CosyVoiceModelFileSpec("llm_config.json", 456L, "81A28DA4C94A85273C5694D85499F760DC37B36766E484AF0581533B641BE41D"),
            CosyVoiceModelFileSpec("llm.mnn", 419_640L, "939C364630A8B61E7ACFBDBDA65FA4D0DDC973E751CEF6C2EC84A1E3BB918BAD"),
            CosyVoiceModelFileSpec("llm.mnn.weight", 352_844_666L, "325948C3809FBA1A98C19AA223537D7D984E5DB206BF165A33CC6AEF80470CA3"),
            CosyVoiceModelFileSpec("embeddings_bf16.bin", 284_426_240L, "FF8FBF42A814A2454AAD569218C0DFB271934D9DD14F237F48DEA08124738BB3"),
            CosyVoiceModelFileSpec("tokenizer.mtok", 4_042_638L, "90C5707092C11E2C1E34C3325C2E92ED6E8FFFCAFB1170E67A0DCA459CFB8C70"),
            CosyVoiceModelFileSpec("flow.cfg-student-2step.batch1.fp16.mnn", 1_340_608L, "C70084684EC71908C38FAD043B07534F3BCF1999810353361405A0564F14C1EC"),
            CosyVoiceModelFileSpec("flow.cfg-student-2step.batch1.fp16.mnn.weight", 663_253_312L, "E8D363544063702A9DA0800F3A2BBBA5669EA719943A5DE15A8B08B0F28D24BC"),
            CosyVoiceModelFileSpec("flow-conditioner.fp32.mnn", 4_403_404L, "E46A28D7E76BDB1C15703549BA9A8AC9C0E090D7FC2B264AD84D26909B65E6AF"),
            CosyVoiceModelFileSpec("prompt-speech-tokens.csv", 400L, "58E3AB5495425691F353C094B387FBE65EBD93509DADA0F1F3F30CFFD9D3B6A3"),
            CosyVoiceModelFileSpec("prompt-cond.bin", 55_680L, "1F7ADD933FA2490FE4F27138C436199495770382AD7CFF29CA9D74815C98FDFC"),
            CosyVoiceModelFileSpec("spks.bin", 320L, "6FF6DD7D3DF8AB6ECD3DF90721B8451750A89511E09633CA5EA39C01C557B3EA"),
            CosyVoiceModelFileSpec("rand-noise.bin", 4_800_000L, "656C9256457B71D1621F32D64715E922C656670185E70B457F7734E2C4DA0B95"),
            CosyVoiceModelFileSpec("hift-f0.fp32.mnn", 13_271_724L, "884604068193AF8D614A75E06FC54FCE44EB0F5DD743E99929FE058D43E6A951"),
            CosyVoiceModelFileSpec("hift-core.fp32.mnn", 70_218_952L, "6BE4E5CA02DE78348D850A6E1D330C01DA57525387B8BA377749671E6D361C84"),
            CosyVoiceModelFileSpec("source-linear-weight.bin", 36L, "1C338F6F59212BE433346668821360D112731804FEC960D297620C07C73DA6C2"),
            CosyVoiceModelFileSpec("source-linear-bias.bin", 4L, "163A4F7DD40651BAF8565B6B338828FACCBFD20AC0ED05173CE6DD45727F9B13")
        )

        val REQUIRED_MODEL_FILES = MODEL_FILE_SPECS.map { it.name }

        val ENROLLMENT_FILE_SPECS = listOf(
            CosyVoiceModelFileSpec("speech-tokenizer-v3.fp32.inline.mnn", 969_762_652L, "EB436E54A7A4227059F7A61C47DA57EFE2BF1124C89CD24BD2841BBC190A4769"),
            CosyVoiceModelFileSpec("campplus.fp32.mnn", 607_208L, "7F9889B437DABB6A906CF6BC9244C2AA9A0AC6599C602D8ECE31183EA7A5D73D"),
            CosyVoiceModelFileSpec("campplus.fp32.mnn.weight", 27_375_488L, "51D633D3B49AA532080DF1015AC8DF87DA819187C1D9049DE39DAB54E1A4A40B"),
            CosyVoiceModelFileSpec("flow-speaker-affine-weight.bin", 61_440L, "0131B531DDBB1A6120DEE9C0832EDE4D1F76EB73B776F13EC890C522E75350B6"),
            CosyVoiceModelFileSpec("flow-speaker-affine-bias.bin", 320L, "B47EFEAA84822ABE17D7E6308AB9AF189F29A0CB86216D564D9BF4B137B24849")
        )
    }

    private val appContext = context.applicationContext
    private val preferences = appContext.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)
    private val root = File(appContext.filesDir, "cosyvoice3-mnn")
    val modelDir = File(root, "model").apply { mkdirs() }
    val voiceProfilesDir = File(root, "voices").apply { mkdirs() }
    val enrollmentDir = File(root, "enrollment").apply { mkdirs() }
    val gpuCacheDir = File(root, "gpu-cache").apply { mkdirs() }
    val workDir = File(appContext.cacheDir, "cosyvoice3-mnn-work").apply { mkdirs() }

    fun selectedVoiceProfileId(): String = preferences.getString(SELECTED_PROFILE_KEY, DEFAULT_VOICE_PROFILE_ID)
        .orEmpty().ifBlank { DEFAULT_VOICE_PROFILE_ID }

    fun selectVoiceProfile(id: String) {
        voiceProfile(id)
        preferences.edit().putString(SELECTED_PROFILE_KEY, id).apply()
    }

    fun modelStatus(): CosyVoiceModelStatus {
        val missing = MODEL_FILE_SPECS.filter { spec ->
            val file = modelFile(spec)
            !file.exists() || file.length() != spec.bytes
        }.map { it.name }
        return CosyVoiceModelStatus(
            ready = missing.isEmpty(),
            installedBytes = MODEL_FILE_SPECS.sumOf { modelFile(it).length() },
            missingFiles = missing
        )
    }

    fun enrollmentStatus(): CosyVoiceModelStatus {
        val missing = ENROLLMENT_FILE_SPECS.filter { spec ->
            val file = enrollmentFile(spec.name)
            !file.exists() || file.length() != spec.bytes
        }.map { it.name }
        return CosyVoiceModelStatus(
            ready = missing.isEmpty(),
            installedBytes = ENROLLMENT_FILE_SPECS.sumOf { enrollmentFile(it.name).length() },
            missingFiles = missing
        )
    }

    fun importEnrollmentZip(uri: Uri, onProgress: (String) -> Unit = {}) {
        val staging = File(root, "enrollment-staging-${System.currentTimeMillis()}").apply { mkdirs() }
        val accepted = mutableSetOf<String>()
        try {
            appContext.contentResolver.openInputStream(uri)?.use { input ->
                ZipInputStream(input.buffered()).use { zip ->
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        if (entry.isDirectory) continue
                        val name = File(entry.name.replace('\\', '/')).name
                        val spec = ENROLLMENT_FILE_SPECS.firstOrNull { it.name == name } ?: continue
                        onProgress("正在导入音色创建扩展 · $name")
                        val target = File(staging, name)
                        copyZipEntry(zip, target, spec.bytes)
                        verifyFile(target, spec, checkHash = true)
                        accepted += name
                    }
                }
            } ?: error("无法读取音色创建扩展")
            val missing = ENROLLMENT_FILE_SPECS.map { it.name }.toSet() - accepted
            check(missing.isEmpty()) { "音色创建扩展缺少：${missing.joinToString()}" }
            ENROLLMENT_FILE_SPECS.forEach { spec ->
                val source = File(staging, spec.name)
                val temp = File(enrollmentDir, "${spec.name}.importing")
                source.copyTo(temp, overwrite = true)
                verifyFile(temp, spec, checkHash = false)
                enrollmentFile(spec.name).delete()
                check(temp.renameTo(enrollmentFile(spec.name))) { "${spec.name} 无法保存" }
            }
            Log.i(TAG, "enrollment extension import done bytes=${enrollmentStatus().installedBytes}")
        } finally {
            staging.deleteRecursively()
        }
    }

    fun deleteEnrollment() {
        ENROLLMENT_FILE_SPECS.forEach { enrollmentFile(it.name).delete() }
        Log.i(TAG, "enrollment extension deleted")
    }

    fun exportEnrollmentZip(uri: Uri, onProgress: (String) -> Unit = {}) {
        check(enrollmentStatus().ready) { "音色创建扩展不完整，无法导出" }
        val output = appContext.contentResolver.openOutputStream(uri, "w") ?: error("无法创建音色创建扩展包")
        try {
            ZipOutputStream(output.buffered(1024 * 1024)).use { zip ->
                zip.setLevel(Deflater.NO_COMPRESSION)
                ENROLLMENT_FILE_SPECS.forEachIndexed { index, spec ->
                    onProgress("正在导出创建扩展 ${index + 1}/${ENROLLMENT_FILE_SPECS.size} · ${spec.name}")
                    val source = enrollmentFile(spec.name)
                    verifyFile(source, spec, checkHash = false)
                    zip.putNextEntry(ZipEntry(spec.name))
                    source.inputStream().buffered(1024 * 1024).use { it.copyTo(zip, 1024 * 1024) }
                    zip.closeEntry()
                }
            }
            Log.i(TAG, "enrollment extension exported bytes=${enrollmentStatus().installedBytes}")
        } catch (error: Throwable) {
            runCatching { appContext.contentResolver.delete(uri, null, null) }
            throw error
        }
    }

    fun executable(fileName: String): File {
        val file = File(appContext.applicationInfo.nativeLibraryDir, fileName)
        check(file.isFile && file.canExecute()) { "原生程序不可执行：${file.absolutePath}" }
        return file
    }

    fun voiceProfile(id: String = DEFAULT_VOICE_PROFILE_ID): CosyVoiceVoiceProfile {
        check(modelStatus().ready) { "MNN 模型不完整" }
        if (id == DEFAULT_VOICE_PROFILE_ID) ensureDefaultVoiceProfile()
        return loadVoiceProfile(id)
    }

    fun voiceProfiles(): List<CosyVoiceVoiceProfile> {
        if (!modelStatus().ready) return emptyList()
        ensureDefaultVoiceProfile()
        return voiceProfilesDir.listFiles()
            .orEmpty()
            .filter { it.isDirectory && !it.name.startsWith('.') }
            .mapNotNull { directory -> runCatching { loadVoiceProfile(directory.name) }.getOrNull() }
            .sortedWith(compareByDescending<CosyVoiceVoiceProfile> { it.builtIn }.thenBy { it.createdAt }.thenBy { it.displayName })
    }

    fun importVoiceProfileZip(uri: Uri): CosyVoiceVoiceProfile =
        importVoiceProfilesZip(uri).firstOrNull() ?: error("音色包中没有可导入的音色")

    fun importVoiceProfilesZip(uri: Uri): List<CosyVoiceVoiceProfile> {
        check(modelStatus().ready) { "请先导入完整 MNN 模型" }
        val stagingRoot = File(voiceProfilesDir, ".import-${System.currentTimeMillis()}").apply { deleteRecursively(); mkdirs() }
        val allowed = setOf(
            CosyVoiceVoiceProfile.METADATA_FILE, CosyVoiceVoiceProfile.PROMPT_SPEECH_TOKENS_FILE,
            CosyVoiceVoiceProfile.PROMPT_CONDITION_FILE, CosyVoiceVoiceProfile.FLOW_SPEAKER_FILE,
            CosyVoiceVoiceProfile.SOURCE_AUDIO_FILE
        )
        val sizeLimits = mapOf(
            CosyVoiceVoiceProfile.METADATA_FILE to 64L * 1024L,
            CosyVoiceVoiceProfile.PROMPT_SPEECH_TOKENS_FILE to 16L * 1024L,
            CosyVoiceVoiceProfile.PROMPT_CONDITION_FILE to 600L * 1024L,
            CosyVoiceVoiceProfile.FLOW_SPEAKER_FILE to 1024L,
            CosyVoiceVoiceProfile.SOURCE_AUDIO_FILE to 30L * 1024L * 1024L
        )
        try {
            appContext.contentResolver.openInputStream(uri)?.use { input ->
                ZipInputStream(input.buffered()).use { zip ->
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        if (entry.isDirectory) continue
                        val parts = entry.name.replace('\\', '/').split('/').filter { it.isNotBlank() }
                        val name = parts.lastOrNull() ?: continue
                        if (name !in allowed) continue
                        val group = when {
                            parts.size == 1 -> "root"
                            parts.size >= 3 && parts.first() == "voices" -> parts[1]
                            else -> parts[parts.lastIndex - 1]
                        }
                        require(group.matches(Regex("[A-Za-z0-9._-]+"))) { "音色包目录名不合法" }
                        val directory = File(stagingRoot, group).apply { mkdirs() }
                        val target = File(directory, name)
                        check(!target.exists()) { "音色包包含重复文件：$group/$name" }
                        copyZipEntry(zip, target, sizeLimits.getValue(name))
                    }
                }
            } ?: error("无法读取音色包")
            val candidates = stagingRoot.listFiles().orEmpty().filter { File(it, CosyVoiceVoiceProfile.METADATA_FILE).isFile }
            check(candidates.isNotEmpty()) { "音色包中没有 profile.json" }
            return candidates.map(::installImportedVoiceProfile)
        } finally {
            stagingRoot.deleteRecursively()
        }
    }

    private fun installImportedVoiceProfile(staging: File): CosyVoiceVoiceProfile {
        val required = setOf(CosyVoiceVoiceProfile.METADATA_FILE, CosyVoiceVoiceProfile.PROMPT_SPEECH_TOKENS_FILE,
            CosyVoiceVoiceProfile.PROMPT_CONDITION_FILE, CosyVoiceVoiceProfile.FLOW_SPEAKER_FILE)
        val missing = required.filterNot { File(staging, it).isFile }
        check(missing.isEmpty()) { "音色包缺少：${missing.joinToString()}" }
        val imported = CosyVoiceVoiceProfile.fromJson(
            JSONObject(File(staging, CosyVoiceVoiceProfile.METADATA_FILE).readText(Charsets.UTF_8)), staging)
        require(imported.id.matches(Regex("[A-Za-z0-9._-]+"))) { "音色档案 ID 不合法" }
        check(imported.modelId == MODEL_ID) { "音色档案与当前模型不兼容" }
        check(!imported.builtIn) { "外部音色不能标记为内置音色" }
        val profileId = nextAvailableProfileId(imported.id)
        Os.symlink(modelFile(CosyVoiceVoiceProfile.SHARED_NOISE_FILE).absolutePath,
            File(staging, CosyVoiceVoiceProfile.SHARED_NOISE_FILE).absolutePath)
        val profile = imported.copy(id = profileId,
            profileHash = profileHash(imported.promptPrefix, staging),
            builtIn = false, createdAt = System.currentTimeMillis(), directory = staging)
        File(staging, CosyVoiceVoiceProfile.METADATA_FILE).writeText(profile.toJson().toString(2), Charsets.UTF_8)
        validateVoiceProfile(profile)
        val target = File(voiceProfilesDir, profileId)
        check(staging.renameTo(target)) { "音色档案无法保存" }
        return loadVoiceProfile(profileId).also {
            Log.i(TAG, "voice profile imported id=${it.id} tokens=${it.promptTokenCount} hash=${it.profileHash}")
        }
    }

    fun deleteVoiceProfile(id: String) {
        val profile = loadVoiceProfile(id)
        check(!profile.builtIn) { "内置音色不能删除" }
        check(profile.directory.deleteRecursively()) { "音色档案删除失败" }
        if (selectedVoiceProfileId() == id) {
            preferences.edit().putString(SELECTED_PROFILE_KEY, DEFAULT_VOICE_PROFILE_ID).apply()
        }
        Log.i(TAG, "voice profile deleted id=$id")
    }

    fun createRealtimeVoiceProfile(id: String): CosyVoiceVoiceProfile {
        val source = loadVoiceProfile(id)
        check(!source.builtIn) { "内置音色已经是实时长度" }
        check(source.promptTokenCount > MAX_REALTIME_PROMPT_TOKENS) { "当前音色无需优化" }
        val targetId = nextAvailableProfileId("${source.id}-realtime")
        val staging = File(voiceProfilesDir, ".$targetId").apply { deleteRecursively(); mkdirs() }
        try {
            val tokens = source.promptSpeechTokensFile.readText(Charsets.US_ASCII)
                .split(',').map { it.trim() }.filter { it.isNotEmpty() }.take(MAX_REALTIME_PROMPT_TOKENS)
            File(staging, CosyVoiceVoiceProfile.PROMPT_SPEECH_TOKENS_FILE).writeText(tokens.joinToString(","), Charsets.US_ASCII)
            val newFrames = MAX_REALTIME_PROMPT_TOKENS * 2
            val oldCondition = source.promptConditionFile.readBytes()
            val newCondition = ByteArray(80 * newFrames * 4)
            val oldChannelBytes = source.promptFrameCount * 4
            val newChannelBytes = newFrames * 4
            repeat(80) { channel ->
                oldCondition.copyInto(newCondition, channel * newChannelBytes, channel * oldChannelBytes,
                    channel * oldChannelBytes + newChannelBytes)
            }
            File(staging, CosyVoiceVoiceProfile.PROMPT_CONDITION_FILE).writeBytes(newCondition)
            source.flowSpeakerFile.copyTo(File(staging, CosyVoiceVoiceProfile.FLOW_SPEAKER_FILE), overwrite = true)
            File(source.directory, CosyVoiceVoiceProfile.SOURCE_AUDIO_FILE).takeIf(File::isFile)
                ?.copyTo(File(staging, CosyVoiceVoiceProfile.SOURCE_AUDIO_FILE), overwrite = true)
            Os.symlink(modelFile(CosyVoiceVoiceProfile.SHARED_NOISE_FILE).absolutePath,
                File(staging, CosyVoiceVoiceProfile.SHARED_NOISE_FILE).absolutePath)
            val sourceText = source.promptPrefix.removePrefix(SYSTEM_PROMPT_PREFIX)
            val textRatio = MAX_REALTIME_PROMPT_TOKENS.toDouble() / source.promptTokenCount
            val realtimeText = sourceText.take((sourceText.length * textRatio).toInt().coerceAtLeast(1))
            val promptPrefix = SYSTEM_PROMPT_PREFIX + realtimeText
            val profile = CosyVoiceVoiceProfile(
                schemaVersion = VOICE_PROFILE_SCHEMA, id = targetId,
                displayName = "${source.displayName}（实时）".take(80), modelId = MODEL_ID,
                promptPrefix = promptPrefix, promptTokenCount = MAX_REALTIME_PROMPT_TOKENS,
                promptFrameCount = newFrames, profileHash = profileHash(promptPrefix, staging),
                builtIn = false, createdAt = System.currentTimeMillis(), directory = staging)
            File(staging, CosyVoiceVoiceProfile.METADATA_FILE).writeText(profile.toJson().toString(2), Charsets.UTF_8)
            validateVoiceProfile(profile)
            val target = File(voiceProfilesDir, targetId)
            check(staging.renameTo(target)) { "实时音色档案无法保存" }
            return loadVoiceProfile(targetId).also {
                Log.i(TAG, "realtime voice profile created source=$id id=${it.id} tokens=${it.promptTokenCount}")
            }
        } finally { staging.deleteRecursively() }
    }

    fun exportVoiceProfilesZip(uri: Uri, profileIds: Collection<String>, onProgress: (String) -> Unit = {}) {
        val profiles = profileIds.distinct().map(::loadVoiceProfile)
        check(profiles.isNotEmpty()) { "没有可导出的音色" }
        val output = appContext.contentResolver.openOutputStream(uri, "w") ?: error("无法创建音色包")
        try {
            ZipOutputStream(output.buffered()).use { zip ->
                val batch = profiles.size > 1
                if (batch) {
                    zip.putNextEntry(ZipEntry("manifest.json"))
                    zip.write(JSONObject().put("schemaVersion", 1).put("type", "cosyvoice3-voice-collection")
                        .put("count", profiles.size).toString(2).toByteArray(Charsets.UTF_8))
                    zip.closeEntry()
                }
                profiles.forEachIndexed { index, profile ->
                    onProgress("正在导出音色 ${index + 1}/${profiles.size} · ${profile.displayName}")
                    val prefix = if (batch) "voices/${profile.id}/" else ""
                    listOf(CosyVoiceVoiceProfile.METADATA_FILE, CosyVoiceVoiceProfile.PROMPT_SPEECH_TOKENS_FILE,
                        CosyVoiceVoiceProfile.PROMPT_CONDITION_FILE, CosyVoiceVoiceProfile.FLOW_SPEAKER_FILE,
                        CosyVoiceVoiceProfile.SOURCE_AUDIO_FILE).forEach { name ->
                        val source = File(profile.directory, name)
                        if (!source.isFile) return@forEach
                        zip.putNextEntry(ZipEntry(prefix + name))
                        source.inputStream().use { it.copyTo(zip) }
                        zip.closeEntry()
                    }
                }
            }
            Log.i(TAG, "voice profiles exported count=${profiles.size}")
        } catch (error: Throwable) {
            runCatching { appContext.contentResolver.delete(uri, null, null) }
            throw error
        }
    }

    fun installEnrolledVoiceProfile(
        displayName: String, promptText: String, nativeOutputDirectory: File, sourceWav: File
    ): CosyVoiceVoiceProfile {
        check(modelStatus().ready) { "MNN 朗读模型不完整" }
        val cleanName = displayName.trim()
        val cleanText = promptText.replace(Regex("[\\r\\n]+"), " ").replace("<|endofprompt|>", "").trim()
        require(cleanName.isNotBlank()) { "请填写音色名称" }
        require(cleanText.isNotBlank()) { "请填写参考音频对应文字" }
        val tokenSource = File(nativeOutputDirectory, CosyVoiceVoiceProfile.PROMPT_SPEECH_TOKENS_FILE)
        val conditionSource = File(nativeOutputDirectory, CosyVoiceVoiceProfile.PROMPT_CONDITION_FILE)
        val speakerSource = File(nativeOutputDirectory, CosyVoiceVoiceProfile.FLOW_SPEAKER_FILE)
        val tokenValues = readVoiceTokens(tokenSource)
        require(tokenValues.size in 1..MAX_REALTIME_PROMPT_TOKENS) { "音色提示 Token 过长：${tokenValues.size}，最多 $MAX_REALTIME_PROMPT_TOKENS" }
        validateTokenValues(tokenValues)
        val tokens = tokenValues.size
        val frames = tokens * 2
        check(conditionSource.length() == 80L * frames * 4L) { "音色提示 Mel 文件不完整" }
        check(speakerSource.length() == 80L * 4L) { "音色说话人文件不完整" }
        check(sourceWav.isFile && sourceWav.length() > 44L) { "参考音频文件不完整" }
        val id = nextAvailableProfileId("voice-${System.currentTimeMillis()}")
        val staging = File(voiceProfilesDir, ".$id").apply { deleteRecursively(); mkdirs() }
        try {
            tokenSource.copyTo(File(staging, CosyVoiceVoiceProfile.PROMPT_SPEECH_TOKENS_FILE))
            conditionSource.copyTo(File(staging, CosyVoiceVoiceProfile.PROMPT_CONDITION_FILE))
            speakerSource.copyTo(File(staging, CosyVoiceVoiceProfile.FLOW_SPEAKER_FILE))
            sourceWav.copyTo(File(staging, CosyVoiceVoiceProfile.SOURCE_AUDIO_FILE))
            Os.symlink(modelFile(CosyVoiceVoiceProfile.SHARED_NOISE_FILE).absolutePath,
                File(staging, CosyVoiceVoiceProfile.SHARED_NOISE_FILE).absolutePath)
            val promptPrefix = SYSTEM_PROMPT_PREFIX + cleanText
            val profile = CosyVoiceVoiceProfile(
                schemaVersion = VOICE_PROFILE_SCHEMA, id = id, displayName = cleanName.take(80),
                modelId = MODEL_ID, promptPrefix = promptPrefix, promptTokenCount = tokens,
                promptFrameCount = frames, profileHash = profileHash(promptPrefix, staging),
                builtIn = false, createdAt = System.currentTimeMillis(), directory = staging)
            File(staging, CosyVoiceVoiceProfile.METADATA_FILE).writeText(profile.toJson().toString(2), Charsets.UTF_8)
            validateVoiceProfile(profile)
            val target = File(voiceProfilesDir, id)
            check(staging.renameTo(target)) { "音色档案无法保存" }
            return loadVoiceProfile(id).also {
                Log.i(TAG, "voice enrolled id=${it.id} tokens=${it.promptTokenCount} hash=${it.profileHash}")
            }
        } finally { staging.deleteRecursively() }
    }

    fun importModelZip(uri: Uri, onProgress: (String) -> Unit = {}) {
        val staging = File(root, "model-staging-${System.currentTimeMillis()}").apply { mkdirs() }
        val accepted = mutableSetOf<String>()
        try {
            appContext.contentResolver.openInputStream(uri)?.use { input ->
                ZipInputStream(input.buffered()).use { zip ->
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        if (entry.isDirectory) continue
                        val name = File(entry.name.replace('\\', '/')).name
                        val spec = MODEL_FILE_SPECS.firstOrNull { it.name == name } ?: continue
                        onProgress("正在导入 $name")
                        val target = File(staging, name)
                        FileOutputStream(target).buffered().use { output -> zip.copyTo(output, 1024 * 1024) }
                        verifyModelFile(target, spec, checkHash = true)
                        accepted += name
                    }
                }
            } ?: error("无法读取模型包")
            val missing = REQUIRED_MODEL_FILES - accepted
            check(missing.isEmpty()) { "模型包缺少：${missing.joinToString()}" }
            MODEL_FILE_SPECS.forEach { spec ->
                val source = File(staging, spec.name)
                val temp = File(modelDir, "${spec.name}.importing")
                source.copyTo(temp, overwrite = true)
                verifyModelFile(temp, spec, checkHash = false)
                modelFile(spec).delete()
                check(temp.renameTo(modelFile(spec))) { "${spec.name} 无法保存" }
            }
            Log.i(TAG, "model import done bytes=${modelStatus().installedBytes}")
        } finally { staging.deleteRecursively() }
    }

    fun exportModelZip(uri: Uri, onProgress: (String) -> Unit = {}) {
        check(modelStatus().ready) { "MNN 模型不完整，无法导出" }
        val output = appContext.contentResolver.openOutputStream(uri, "w") ?: error("无法创建模型包")
        var exportedBytes = 0L
        val totalBytes = MODEL_FILE_SPECS.sumOf { it.bytes }
        var lastProgress = -1L
        try {
            ZipOutputStream(output.buffered(1024 * 1024)).use { zip ->
                zip.setLevel(Deflater.NO_COMPRESSION)
                val buffer = ByteArray(1024 * 1024)
                MODEL_FILE_SPECS.forEach { spec ->
                    val source = modelFile(spec)
                    verifyModelFile(source, spec, checkHash = false)
                    zip.putNextEntry(ZipEntry(spec.name))
                    source.inputStream().buffered(1024 * 1024).use { input ->
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            zip.write(buffer, 0, count)
                            exportedBytes += count
                            val progress = exportedBytes * 100L / totalBytes
                            if (progress != lastProgress) { lastProgress = progress; onProgress("正在导出 ${spec.name} · $progress%") }
                        }
                    }
                    zip.closeEntry()
                }
            }
            Log.i(TAG, "model export done bytes=$exportedBytes")
        } catch (error: Throwable) {
            runCatching { appContext.contentResolver.delete(uri, null, null) }
            throw error
        }
    }

    fun llmRuntimeConfig(): File {
        val source = modelFile("config-cpu-cosyvoice-ras.json")
        check(source.isFile) { "缺少 LLM 配置文件" }
        val target = File(modelDir, "config-cpu-cosyvoice-ras-runtime.json")
        val content = JSONObject(source.readText(Charsets.UTF_8)).put("thread_num", 4).put("power", "high").toString(2)
        if (!target.isFile || target.readText(Charsets.UTF_8) != content) target.writeText(content, Charsets.UTF_8)
        return target
    }

    fun deleteModel() {
        MODEL_FILE_SPECS.forEach { spec ->
            modelFile(spec).delete(); modelPartFile(spec).delete(); modelVerifiedFile(spec).delete()
        }
        File(modelDir, "config-cpu-cosyvoice-ras-runtime.json").delete()
        gpuCacheDir.listFiles()?.forEach(File::delete)
        Log.i(TAG, "model deleted")
    }

    private fun ensureDefaultVoiceProfile() {
        val profileHash = defaultProfileHash()
        val existing = runCatching { loadVoiceProfile(DEFAULT_VOICE_PROFILE_ID) }.getOrNull()
        if (existing?.profileHash == profileHash) return
        val staging = File(voiceProfilesDir, ".$DEFAULT_VOICE_PROFILE_ID-${System.currentTimeMillis()}").apply { deleteRecursively(); mkdirs() }
        try {
            listOf(CosyVoiceVoiceProfile.PROMPT_SPEECH_TOKENS_FILE, CosyVoiceVoiceProfile.PROMPT_CONDITION_FILE,
                CosyVoiceVoiceProfile.FLOW_SPEAKER_FILE).forEach { name ->
                modelFile(name).copyTo(File(staging, name), overwrite = true)
            }
            Os.symlink(modelFile(CosyVoiceVoiceProfile.SHARED_NOISE_FILE).absolutePath,
                File(staging, CosyVoiceVoiceProfile.SHARED_NOISE_FILE).absolutePath)
            val profile = CosyVoiceVoiceProfile(
                schemaVersion = VOICE_PROFILE_SCHEMA, id = DEFAULT_VOICE_PROFILE_ID,
                displayName = FIXED_VOICE_NAME, modelId = MODEL_ID, promptPrefix = PROMPT_PREFIX,
                promptTokenCount = DEFAULT_PROMPT_TOKENS, promptFrameCount = DEFAULT_PROMPT_FRAMES,
                profileHash = profileHash, builtIn = true, createdAt = System.currentTimeMillis(), directory = staging)
            File(staging, CosyVoiceVoiceProfile.METADATA_FILE).writeText(profile.toJson().toString(2), Charsets.UTF_8)
            validateVoiceProfile(profile)
            val profileDirectory = File(voiceProfilesDir, DEFAULT_VOICE_PROFILE_ID)
            profileDirectory.deleteRecursively()
            check(staging.renameTo(profileDirectory)) { "默认音色档案无法保存" }
            Log.i(TAG, "default voice profile migrated id=$DEFAULT_VOICE_PROFILE_ID hash=$profileHash")
        } finally { staging.deleteRecursively() }
    }

    private fun loadVoiceProfile(id: String): CosyVoiceVoiceProfile {
        require(id.matches(Regex("[A-Za-z0-9._-]+"))) { "音色档案 ID 不合法" }
        val directory = File(voiceProfilesDir, id)
        val metadata = File(directory, CosyVoiceVoiceProfile.METADATA_FILE)
        check(metadata.isFile) { "音色档案不存在：$id" }
        val profile = CosyVoiceVoiceProfile.fromJson(JSONObject(metadata.readText(Charsets.UTF_8)), directory)
        check(profile.id == id) { "音色档案 ID 不一致" }
        check(profile.modelId == MODEL_ID) { "音色档案与当前模型不兼容" }
        validateVoiceProfile(profile)
        return profile
    }

    private fun validateVoiceProfile(profile: CosyVoiceVoiceProfile) {
        check(profile.schemaVersion == VOICE_PROFILE_SCHEMA) { "音色档案版本不支持" }
        check(profile.promptPrefix.isNotBlank()) { "音色档案缺少提示文本" }
        check(profile.promptTokenCount in 1..375) { "音色 prompt token 数量不正确" }
        check(profile.promptFrameCount == profile.promptTokenCount * 2) { "音色 prompt token 与 Mel 帧不对齐" }
        val tokenValues = readVoiceTokens(profile.promptSpeechTokensFile)
        check(tokenValues.size == profile.promptTokenCount) { "音色 prompt token 文件不完整" }
        validateTokenValues(tokenValues)
        check(profile.promptConditionFile.length() == 80L * profile.promptFrameCount * 4L) { "音色 prompt condition 文件不完整" }
        check(profile.flowSpeakerFile.length() == 80L * 4L) { "音色 speaker 文件不完整" }
        check(profile.sharedNoiseFile.length() == 80L * 15_000L * 4L) { "共享 Flow noise 文件不可用" }
        check(profile.profileHash.length == 64) { "音色档案哈希不正确" }
    }

    private fun readVoiceTokens(file: File): List<Int> = file.readText(Charsets.US_ASCII)
        .split(',').map { it.trim() }.filter { it.isNotEmpty() }
        .map { value -> value.toIntOrNull() ?: error("音色 prompt token 格式不正确") }

    private fun validateTokenValues(tokens: List<Int>) {
        check(tokens.all { it in 0..6560 }) { "音色 prompt token 超出 CosyVoice3 码本范围" }
        check(tokens.distinct().size >= 4) { "音色 prompt token 无效：缺少语音变化，请重新导入新版音色创建扩展后创建" }
    }

    private fun defaultProfileHash(): String = profileHash(PROMPT_PREFIX, modelDir)

    private fun profileHash(promptPrefix: String, directory: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(MODEL_ID.toByteArray(Charsets.UTF_8))
        digest.update(promptPrefix.toByteArray(Charsets.UTF_8))
        listOf(CosyVoiceVoiceProfile.PROMPT_SPEECH_TOKENS_FILE, CosyVoiceVoiceProfile.PROMPT_CONDITION_FILE,
            CosyVoiceVoiceProfile.FLOW_SPEAKER_FILE).forEach { name ->
            File(directory, name).inputStream().use { input ->
                val buffer = ByteArray(64 * 1024)
                while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun nextAvailableProfileId(requestedId: String): String {
        if (!File(voiceProfilesDir, requestedId).exists()) return requestedId
        for (suffix in 2..999) {
            val candidate = "$requestedId-$suffix"
            if (!File(voiceProfilesDir, candidate).exists()) return candidate
        }
        error("同名音色档案过多")
    }

    private fun copyZipEntry(zip: ZipInputStream, target: File, maxBytes: Long) {
        FileOutputStream(target).buffered().use { output ->
            val buffer = ByteArray(64 * 1024)
            var copied = 0L
            while (true) {
                val count = zip.read(buffer)
                if (count < 0) break
                copied += count
                check(copied <= maxBytes) { "${target.name} 超过允许大小" }
                output.write(buffer, 0, count)
            }
        }
    }

    fun modelFile(spec: CosyVoiceModelFileSpec): File = File(modelDir, spec.name)
    fun modelFile(name: String): File = File(modelDir, name)
    fun enrollmentFile(name: String): File = File(enrollmentDir, name)
    fun modelPartFile(spec: CosyVoiceModelFileSpec): File = File(modelDir, "${spec.name}.download")
    fun modelVerifiedFile(spec: CosyVoiceModelFileSpec): File = File(modelDir, "${spec.name}.verified")

    fun verifyModelFile(file: File, spec: CosyVoiceModelFileSpec, checkHash: Boolean) = verifyFile(file, spec, checkHash)

    private fun verifyFile(file: File, spec: CosyVoiceModelFileSpec, checkHash: Boolean) {
        check(file.exists() && file.length() == spec.bytes) {
            "${spec.name} 大小不正确：${file.length()} / ${spec.bytes}"
        }
        if (checkHash) {
            val actual = sha256(file)
            check(actual.equals(spec.sha256, ignoreCase = true)) { "${spec.name} 校验失败，请重新导入" }
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered(1024 * 1024).use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
        }
        return digest.digest().joinToString("") { "%02X".format(it) }
    }
}
