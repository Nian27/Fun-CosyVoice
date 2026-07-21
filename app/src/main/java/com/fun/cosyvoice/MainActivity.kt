package com.fun.cosyvoice

import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import org.json.JSONObject

private data class VoiceEnrollmentRequest(
    val displayName: String,
    val promptText: String,
    val startSeconds: Double,
    val endSeconds: Double
)

class MainActivity : ComponentActivity() {

    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private val runtimeReleaseScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val store by lazy { CosyVoiceStore(this) }
    private var modelStatus by mutableStateOf(CosyVoiceModelStatus(false, 0L, emptyList()))
    private var enrollmentStatus by mutableStateOf(CosyVoiceModelStatus(false, 0L, emptyList()))
    private var voiceProfiles by mutableStateOf<List<CosyVoiceVoiceProfile>>(emptyList())
    private var selectedVoiceProfileId by mutableStateOf(CosyVoiceStore.DEFAULT_VOICE_PROFILE_ID)
    private var busyMessage by mutableStateOf("")
    private var resultMessage by mutableStateOf("")
    private var selectedAudioUri by mutableStateOf<Uri?>(null)
    private var selectedAudioName by mutableStateOf("")
    private var mediaPlayer: MediaPlayer? = null
    private var pendingVoiceExportIds: List<String> = emptyList()

    private val modelZipLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(::importModel)
    }
    private val modelExportLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        uri?.let(::exportModel)
    }
    private val voiceProfileZipLauncher = registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) importVoiceProfiles(uris)
    }
    private val voiceProfileExportLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        val ids = pendingVoiceExportIds
        pendingVoiceExportIds = emptyList()
        if (uri != null && ids.isNotEmpty()) exportVoiceProfiles(uri, ids)
    }
    private val enrollmentZipLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(::importEnrollmentExtension)
    }
    private val enrollmentExportLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        uri?.let(::exportEnrollmentExtension)
    }
    private val referenceAudioLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            runCatching { contentResolver.takePersistableUriPermission(uri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION) }
            selectedAudioUri = uri
            selectedAudioName = displayName(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        selectedVoiceProfileId = store.selectedVoiceProfileId()
        refresh()
        setContentView(ComposeView(this).apply {
            setContent {
                CosyVoiceManagerScreen(
                    modelStatus, enrollmentStatus, voiceProfiles, selectedVoiceProfileId,
                    selectedAudioName, busyMessage, resultMessage,
                    onBack = { finish() },
                    onImportModel = { modelZipLauncher.launch(arrayOf("application/zip", "application/octet-stream")) },
                    onExportModel = { modelExportLauncher.launch("cosyvoice3-mnn-complete.zip") },
                    onDeleteModel = { deleteModel() },
                    onImportVoiceProfile = { voiceProfileZipLauncher.launch(arrayOf("application/zip", "application/octet-stream")) },
                    onExportVoiceProfile = { profile -> startVoiceExport(listOf(profile.id), "${profile.displayName}.zip") },
                    onExportAllVoiceProfiles = {
                        val ids = voiceProfiles.filterNot { it.builtIn }.map { it.id }
                        startVoiceExport(ids, "cosyvoice3-voices.zip")
                    },
                    onVoiceProfileSelected = { selectVoiceProfile(it) },
                    onCreateRealtimeVoiceProfile = { createRealtimeVoiceProfile(it) },
                    onDeleteVoiceProfile = { deleteVoiceProfile(it) },
                    onImportEnrollment = { enrollmentZipLauncher.launch(arrayOf("application/zip", "application/octet-stream")) },
                    onExportEnrollment = { enrollmentExportLauncher.launch("cosyvoice3-mnn-enrollment-extension.zip") },
                    onDeleteEnrollment = { deleteEnrollmentExtension() },
                    onSelectReferenceAudio = { referenceAudioLauncher.launch(arrayOf("audio/*")) },
                    onCreateVoice = { createVoiceProfile(it) },
                    onPreview = { text, options -> preview(text, options) }
                )
            }
        })
    }

    override fun onDestroy() {
        mediaPlayer?.release()
        scope.cancel()
        runtimeReleaseScope.launch {
            try { CosyVoiceRuntime.close() } finally { runtimeReleaseScope.cancel() }
        }
        super.onDestroy()
    }

    private fun refresh() {
        modelStatus = store.modelStatus()
        enrollmentStatus = store.enrollmentStatus()
        voiceProfiles = if (modelStatus.ready) store.voiceProfiles() else emptyList()
        if (voiceProfiles.none { it.id == selectedVoiceProfileId }) {
            selectedVoiceProfileId = CosyVoiceStore.DEFAULT_VOICE_PROFILE_ID
            store.selectVoiceProfile(selectedVoiceProfileId)
        }
    }

    private fun selectVoiceProfile(id: String) { store.selectVoiceProfile(id); selectedVoiceProfileId = id }

    private fun importEnrollmentExtension(uri: Uri) {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在检查音色创建扩展"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) { store.importEnrollmentZip(uri) { message -> scope.launch { busyMessage = message } } }
            }.onSuccess { refresh(); resultMessage = "音色创建扩展导入完成 · ${formatBytes(enrollmentStatus.installedBytes)}" }
                .onFailure { resultMessage = "音色创建扩展导入失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "enrollment import failed", it) }
            busyMessage = ""
        }
    }

    private fun createVoiceProfile(request: VoiceEnrollmentRequest) {
        if (busyMessage.isNotBlank()) return
        val duration = request.endSeconds - request.startSeconds
        if (duration !in 3.0..5.0) { resultMessage = "当前片段 %.1f 秒，请调整为 3-5 秒后再创建".format(duration); return }
        val uri = selectedAudioUri
        if (uri == null) { resultMessage = "请先选择 MP3 或其他音频文件"; return }
        if (!modelStatus.ready || !enrollmentStatus.ready) { resultMessage = "请先导入 MNN 朗读模型和音色创建扩展"; return }
        scope.launch {
            busyMessage = "正在释放朗读模型，为音色注册腾出内存"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) {
                    CosyVoiceRuntime.close()
                    val runDirectory = File(store.workDir, "enroll-${System.currentTimeMillis()}").apply { deleteRecursively(); mkdirs() }
                    try {
                        val sourceWav = File(runDirectory, "source.wav")
                        withContext(Dispatchers.Main.immediate) { busyMessage = "正在解码并截取参考音频" }
                        val decoded = CosyVoiceAudioDecoder.decodeSegmentToWav(this@MainActivity, uri, request.startSeconds, request.endSeconds, sourceWav)
                        val nativeOutput = File(runDirectory, "profile").apply { mkdirs() }
                        withContext(Dispatchers.Main.immediate) { busyMessage = "正在提取语音 Token 和说话人特征" }
                        val exitCode = CosyVoiceEnrollmentNative.enroll(
                            tokenizerModelPath = store.enrollmentFile("speech-tokenizer-v3.fp32.inline.mnn").absolutePath,
                            campPlusModelPath = store.enrollmentFile("campplus.fp32.mnn").absolutePath,
                            affineWeightPath = store.enrollmentFile("flow-speaker-affine-weight.bin").absolutePath,
                            affineBiasPath = store.enrollmentFile("flow-speaker-affine-bias.bin").absolutePath,
                            sourceWavPath = sourceWav.absolutePath,
                            outputDirectory = nativeOutput.absolutePath, threads = 6)
                        check(exitCode == 0) { "${CosyVoiceEnrollmentNative.errorMessage(exitCode)}($exitCode)" }
                        val report = JSONObject(File(nativeOutput, "enrollment-report.json").readText())
                        val profile = store.installEnrolledVoiceProfile(request.displayName, request.promptText, nativeOutput, sourceWav)
                        Triple(profile, decoded, report)
                    } finally { runDirectory.deleteRecursively() }
                }
            }.onSuccess { (profile, decoded, report) ->
                refresh(); selectVoiceProfile(profile.id)
                resultMessage = buildString {
                    append("音色创建完成：${profile.displayName}")
                    append(" · %.2f 秒 · %d Token".format(decoded.durationSeconds, profile.promptTokenCount))
                    append("\n注册耗时 %.2f 秒；已自动选中，可直接合成试听。".format(report.optDouble("totalMs") / 1000.0))
                }
            }.onFailure { resultMessage = "音色创建失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "voice enrollment failed", it) }
            busyMessage = ""
        }
    }

    private fun deleteEnrollmentExtension() {
        if (busyMessage.isNotBlank()) return
        android.app.AlertDialog.Builder(this)
            .setTitle("删除音色创建扩展？").setMessage("将删除约 490 MB 的注册模型。已创建的音色和朗读模型都会保留。")
            .setNegativeButton("取消", null).setPositiveButton("确认删除") { _, _ -> store.deleteEnrollment(); refresh(); resultMessage = "音色创建扩展已删除，现有音色仍可继续使用" }.show()
    }

    private fun exportEnrollmentExtension(uri: Uri) {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在准备导出音色创建扩展"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) { store.exportEnrollmentZip(uri) { message -> scope.launch { busyMessage = message } } }
            }.onSuccess { resultMessage = "音色创建扩展已导出，可在本页重新导入" }
                .onFailure { resultMessage = "创建扩展导出失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "enrollment export failed", it) }
            busyMessage = ""
        }
    }

    private fun displayName(uri: Uri): String {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) return cursor.getString(0).orEmpty()
        }
        return uri.lastPathSegment.orEmpty().ifBlank { "已选择音频" }
    }

    private fun importVoiceProfiles(uris: List<Uri>) {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在检查零样本音色档案"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) {
                    uris.flatMapIndexed { index, uri ->
                        withContext(Dispatchers.Main.immediate) { busyMessage = "正在导入音色包 ${index + 1}/${uris.size}" }
                        store.importVoiceProfilesZip(uri)
                    }
                }
            }.onSuccess { profiles ->
                refresh(); profiles.lastOrNull()?.let { selectVoiceProfile(it.id) }
                resultMessage = "已导入 ${profiles.size} 个音色" + profiles.lastOrNull()?.let { "，当前：${it.displayName}" }.orEmpty()
            }.onFailure { resultMessage = "音色导入失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "voice profile import failed", it) }
            busyMessage = ""
        }
    }

    private fun startVoiceExport(ids: List<String>, fileName: String) {
        if (busyMessage.isNotBlank()) return
        if (ids.isEmpty()) { resultMessage = "没有可导出的自定义音色"; return }
        pendingVoiceExportIds = ids; voiceProfileExportLauncher.launch(fileName.replace(Regex("[\\/:*?\"<>|]"), "_"))
    }

    private fun exportVoiceProfiles(uri: Uri, ids: List<String>) {
        scope.launch {
            busyMessage = "正在准备导出音色"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) { store.exportVoiceProfilesZip(uri, ids) { message -> scope.launch { busyMessage = message } } }
            }.onSuccess { resultMessage = "已导出 ${ids.size} 个音色，可在本页批量导入" }
                .onFailure { resultMessage = "音色导出失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "voice profile export failed", it) }
            busyMessage = ""
        }
    }

    private fun createRealtimeVoiceProfile(profile: CosyVoiceVoiceProfile) {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在生成 125 Token 实时音色"; resultMessage = ""
            runCatching { withContext(Dispatchers.IO) { store.createRealtimeVoiceProfile(profile.id) } }
                .onSuccess { created -> refresh(); selectVoiceProfile(created.id); resultMessage = "实时音色已生成并选中：${created.displayName} · ${created.promptTokenCount} Token" }
                .onFailure { resultMessage = "实时音色生成失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "realtime voice profile failed", it) }
            busyMessage = ""
        }
    }

    private fun importModel(uri: Uri) {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在检查 MNN 模型包"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) { store.importModelZip(uri) { message -> scope.launch { busyMessage = message } } }
            }.onSuccess { refresh(); resultMessage = "MNN 模型导入完成" }
                .onFailure { resultMessage = "模型导入失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "model import failed", it) }
            busyMessage = ""
        }
    }

    private fun preview(text: String, options: CosyVoiceSynthesisOptions) {
        if (busyMessage.isNotBlank()) return
        if (!modelStatus.ready) { resultMessage = "请先导入完整 MNN 模型包"; return }
        val cleanText = text.trim()
        if (cleanText.isBlank()) { resultMessage = "请输入试听文字"; return }
        scope.launch {
            mediaPlayer?.release(); mediaPlayer = null
            busyMessage = "正在启动 MNN 单句链路"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) {
                    val output = File(cacheDir, "cosyvoice-preview/${cleanText.hashCode()}-${options.hashCode()}.wav")
                    CosyVoiceRuntime.synthesize(store, cleanText, output, options) { stage ->
                        withContext(Dispatchers.Main.immediate) { busyMessage = stage }
                    }
                }
            }.onSuccess { report ->
                resultMessage = report.displayText()
                mediaPlayer = MediaPlayer().apply {
                    setDataSource(report.output.absolutePath)
                    setOnPreparedListener { it.start() }
                    setOnCompletionListener { it.release(); if (mediaPlayer === it) mediaPlayer = null }
                    setOnErrorListener { player, what, extra -> resultMessage = "播放失败：$what/$extra"; player.release(); if (mediaPlayer === player) mediaPlayer = null; true }
                    prepareAsync()
                }
            }.onFailure { resultMessage = "试听失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "preview failed", it) }
            busyMessage = ""
        }
    }

    private fun exportModel(uri: Uri) {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在准备导出 MNN 模型"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) { store.exportModelZip(uri) { message -> scope.launch { busyMessage = message } } }
            }.onSuccess { resultMessage = "MNN 模型已导出，可用同一页面的导入 ZIP 恢复" }
                .onFailure { resultMessage = "模型导出失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "model export failed", it) }
            busyMessage = ""
        }
    }

    private fun deleteModel() {
        if (busyMessage.isNotBlank()) return
        android.app.AlertDialog.Builder(this)
            .setTitle("删除本地模型？").setMessage("将删除约 1.3 GB 的 CosyVoice3 模型和 GPU 编译缓存。删除后必须重新导入完整 ZIP 才能试听。")
            .setNegativeButton("取消", null).setPositiveButton("确认删除") { _, _ -> performDeleteModel() }.show()
    }

    private fun performDeleteModel() {
        scope.launch {
            busyMessage = "正在删除 MNN 模型"
            withContext(Dispatchers.IO) { CosyVoiceRuntime.close(); store.deleteModel() }
            refresh(); busyMessage = ""; resultMessage = "模型与 GPU 编译缓存已删除"
        }
    }

    private fun deleteVoiceProfile(profile: CosyVoiceVoiceProfile) {
        if (busyMessage.isNotBlank() || profile.builtIn) return
        android.app.AlertDialog.Builder(this)
            .setTitle("删除音色？").setMessage("将删除\"${profile.displayName}\"的本地音色档案。MNN 模型不会被删除。")
            .setNegativeButton("取消", null).setPositiveButton("确认删除") { _, _ ->
                scope.launch {
                    busyMessage = "正在删除音色"
                    runCatching { withContext(Dispatchers.IO) { store.deleteVoiceProfile(profile.id) } }
                        .onSuccess { selectVoiceProfile(CosyVoiceStore.DEFAULT_VOICE_PROFILE_ID); refresh(); resultMessage = "音色已删除：${profile.displayName}" }
                        .onFailure { resultMessage = "音色删除失败：${it.localizedMessage.orEmpty()}" }
                    busyMessage = ""
                }
            }.show()
    }
}

@Composable
private fun CosyVoiceManagerScreen(
    modelStatus: CosyVoiceModelStatus, enrollmentStatus: CosyVoiceModelStatus,
    voiceProfiles: List<CosyVoiceVoiceProfile>, selectedVoiceProfileId: String,
    selectedAudioName: String, busyMessage: String, resultMessage: String,
    onBack: () -> Unit, onImportModel: () -> Unit, onExportModel: () -> Unit, onDeleteModel: () -> Unit,
    onImportVoiceProfile: () -> Unit, onExportVoiceProfile: (CosyVoiceVoiceProfile) -> Unit,
    onExportAllVoiceProfiles: () -> Unit, onVoiceProfileSelected: (String) -> Unit,
    onCreateRealtimeVoiceProfile: (CosyVoiceVoiceProfile) -> Unit, onDeleteVoiceProfile: (CosyVoiceVoiceProfile) -> Unit,
    onImportEnrollment: () -> Unit, onExportEnrollment: () -> Unit, onDeleteEnrollment: () -> Unit,
    onSelectReferenceAudio: () -> Unit, onCreateVoice: (VoiceEnrollmentRequest) -> Unit,
    onPreview: (String, CosyVoiceSynthesisOptions) -> Unit
) {
    var previewText by androidx.compose.runtime.remember { mutableStateOf("你好，欢迎使用阅读，这是手机 MNN 本地合成测试。") }
    var flowGpuMode by androidx.compose.runtime.remember { mutableIntStateOf(68) }
    var hiftBackend by androidx.compose.runtime.remember { mutableStateOf("cpu") }
    var hiftGpuMode by androidx.compose.runtime.remember { mutableIntStateOf(4) }
    var inferenceMode by androidx.compose.runtime.remember { mutableStateOf(CosyVoiceInferenceMode.ZERO_SHOT) }
    var instruction by androidx.compose.runtime.remember { mutableStateOf(CosyVoiceInstruction.DEFAULT_INSTRUCTION) }
    var voiceName by androidx.compose.runtime.remember { mutableStateOf("") }
    var promptText by androidx.compose.runtime.remember { mutableStateOf("") }
    var segmentStart by androidx.compose.runtime.remember { mutableStateOf("0") }
    var segmentEnd by androidx.compose.runtime.remember { mutableStateOf("5") }
    val segmentDuration = segmentStart.toDoubleOrNull()?.let { start -> segmentEnd.toDoubleOrNull()?.minus(start) }

    Column(Modifier.fillMaxSize().statusBarsPadding().background(MaterialTheme.colorScheme.background)) {
        Surface(shadowElevation = 2.dp) {
            Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = onBack) { Text("返回") }
                Text("Fun-CosyVoice 真机测试", fontSize = 20.sp, fontWeight = FontWeight.Bold)
            }
        }
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {

            // --- MNN Model Card ---
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("MNN 模型", fontWeight = FontWeight.Bold, fontSize = 17.sp)
                    Text(
                        if (modelStatus.ready) "已安装 · ${formatBytes(modelStatus.installedBytes)} · 24 kHz 单声道" else "未就绪 · 缺少 ${modelStatus.missingFiles.size} 个文件",
                        color = if (modelStatus.ready) Color(0xFF228B22) else MaterialTheme.colorScheme.error)
                    if (modelStatus.missingFiles.isNotEmpty()) Text(modelStatus.missingFiles.joinToString("\n"), fontSize = 12.sp)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = onImportModel, enabled = busyMessage.isBlank()) { Text("导入 ZIP") }
                        OutlinedButton(onClick = onExportModel, enabled = busyMessage.isBlank() && modelStatus.ready) { Text("导出 ZIP") }
                    }
                    if (modelStatus.installedBytes > 0L) OutlinedButton(onClick = onDeleteModel, enabled = busyMessage.isBlank()) { Text("删除模型") }
                }
            }

            // --- GPU Scheduling Card ---
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("GPU 调度", fontWeight = FontWeight.Bold, fontSize = 17.sp)
                    Text("Flow OpenCL mode", fontSize = 13.sp)
                    ModeButtons(flowGpuMode, busyMessage.isBlank()) { flowGpuMode = it }
                    Text("HiFT core", fontSize = 13.sp)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        ChoiceButton("CPU", hiftBackend == "cpu", busyMessage.isBlank()) { hiftBackend = "cpu" }
                        ChoiceButton("GPU", hiftBackend == "opencl", busyMessage.isBlank()) { hiftBackend = "opencl" }
                    }
                    if (hiftBackend == "opencl") { Text("HiFT OpenCL mode", fontSize = 13.sp); ModeButtons(hiftGpuMode, busyMessage.isBlank()) { hiftGpuMode = it } }
                }
            }

            // --- Voice & Synthesis Card ---
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("音色与演绎", fontWeight = FontWeight.Bold, fontSize = 17.sp)
                    Text("零样本音色", fontSize = 13.sp)
                    voiceProfiles.forEach { profile ->
                        Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                ChoiceButton(profile.displayName, selectedVoiceProfileId == profile.id, busyMessage.isBlank()) { onVoiceProfileSelected(profile.id) }
                                Text("${profile.promptTokenCount} Token", Modifier.weight(1f), fontSize = 12.sp,
                                    color = if (profile.promptTokenCount > CosyVoiceStore.MAX_REALTIME_PROMPT_TOKENS) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            if (!profile.builtIn) {
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                                    if (profile.promptTokenCount > CosyVoiceStore.MAX_REALTIME_PROMPT_TOKENS) TextButton(onClick = { onCreateRealtimeVoiceProfile(profile) }, enabled = busyMessage.isBlank()) { Text("生成实时版") }
                                    TextButton(onClick = { onExportVoiceProfile(profile) }, enabled = busyMessage.isBlank()) { Text("导出") }
                                    TextButton(onClick = { onDeleteVoiceProfile(profile) }, enabled = busyMessage.isBlank()) { Text("删除") }
                                }
                            }
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = onImportVoiceProfile, enabled = busyMessage.isBlank() && modelStatus.ready) { Text("批量导入 ZIP") }
                        OutlinedButton(onClick = onExportAllVoiceProfiles, enabled = busyMessage.isBlank() && voiceProfiles.any { !it.builtIn }) { Text("导出全部") }
                    }

                    Text("手机创建音色", fontWeight = FontWeight.Bold, fontSize = 15.sp)
                    Text(
                        if (enrollmentStatus.ready) "创建扩展已安装 · ${formatBytes(enrollmentStatus.installedBytes)}" else "创建扩展未安装 · 约 490 MB",
                        color = if (enrollmentStatus.ready) Color(0xFF228B22) else MaterialTheme.colorScheme.error, fontSize = 13.sp)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = onImportEnrollment, enabled = busyMessage.isBlank()) { Text("导入创建扩展") }
                        OutlinedButton(onClick = onExportEnrollment, enabled = busyMessage.isBlank() && enrollmentStatus.ready) { Text("导出扩展") }
                    }
                    if (enrollmentStatus.installedBytes > 0L) TextButton(onClick = onDeleteEnrollment, enabled = busyMessage.isBlank()) { Text("删除扩展") }

                    OutlinedButton(onClick = onSelectReferenceAudio, enabled = busyMessage.isBlank()) {
                        Text(if (selectedAudioName.isBlank()) "选择 MP3/音频" else selectedAudioName)
                    }
                    OutlinedTextField(voiceName, { voiceName = it }, label = { Text("音色名称") }, modifier = Modifier.fillMaxWidth(), singleLine = true)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(segmentStart, { segmentStart = it }, label = { Text("起始秒") }, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal), modifier = Modifier.weight(1f), singleLine = true)
                        OutlinedTextField(segmentEnd, { segmentEnd = it }, label = { Text("结束秒") }, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal), modifier = Modifier.weight(1f), singleLine = true)
                    }
                    if (segmentDuration != null) Text("当前片段：%.1f 秒%s".format(segmentDuration, if (segmentDuration in 3.0..5.0) "" else "（需要 3-5 秒）"),
                        color = if (segmentDuration in 3.0..5.0) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.error, fontSize = 12.sp)
                    OutlinedTextField(promptText, { promptText = it }, label = { Text("截取片段对应文字") },
                        supportingText = { Text("实时朗读请选择 3-5 秒清晰单人语音；文字必须与片段完全一致。") },
                        modifier = Modifier.fillMaxWidth(), minLines = 2)
                    Button(
                        onClick = {
                            val start = segmentStart.toDoubleOrNull(); val end = segmentEnd.toDoubleOrNull()
                            if (start != null && end != null) onCreateVoice(VoiceEnrollmentRequest(voiceName, promptText, start, end))
                        },
                        enabled = busyMessage.isBlank() && modelStatus.ready && enrollmentStatus.ready && selectedAudioName.isNotBlank() && voiceName.isNotBlank() && promptText.isNotBlank() && segmentStart.toDoubleOrNull() != null && segmentEnd.toDoubleOrNull() != null,
                        modifier = Modifier.fillMaxWidth()) { Text("创建并选中音色") }

                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        ChoiceButton("普通复刻", inferenceMode == CosyVoiceInferenceMode.ZERO_SHOT, busyMessage.isBlank()) { inferenceMode = CosyVoiceInferenceMode.ZERO_SHOT }
                        ChoiceButton("指令演绎", inferenceMode == CosyVoiceInferenceMode.INSTRUCT2, busyMessage.isBlank()) { inferenceMode = CosyVoiceInferenceMode.INSTRUCT2 }
                    }
                    if (inferenceMode == CosyVoiceInferenceMode.INSTRUCT2) {
                        Text("指令预设", fontSize = 13.sp)
                        CosyVoiceInstruction.presets.chunked(3).forEach { rowPresets ->
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                rowPresets.forEach { preset ->
                                    if (instruction == preset.instruction) Button(onClick = { instruction = preset.instruction }, enabled = busyMessage.isBlank(), modifier = Modifier.weight(1f)) { Text(preset.label) }
                                    else OutlinedButton(onClick = { instruction = preset.instruction }, enabled = busyMessage.isBlank(), modifier = Modifier.weight(1f)) { Text(preset.label) }
                                }
                                repeat(3 - rowPresets.size) { Spacer(Modifier.weight(1f)) }
                            }
                        }
                        OutlinedTextField(instruction, { instruction = it }, label = { Text("演绎指令") },
                            supportingText = { Text("会作为 Instruct2 指令发送给 LLM，不会附加参考语音 Token。") }, modifier = Modifier.fillMaxWidth(), minLines = 2)
                    }

                    OutlinedTextField(previewText, { previewText = it }, label = { Text("试听文字") }, modifier = Modifier.fillMaxWidth(), minLines = 2)
                    Button(
                        onClick = { onPreview(previewText, CosyVoiceSynthesisOptions(flowGpuMode = flowGpuMode, hiftCoreBackend = hiftBackend, hiftGpuMode = hiftGpuMode, voiceProfileId = selectedVoiceProfileId, inferenceMode = inferenceMode, instruction = instruction)) },
                        enabled = busyMessage.isBlank() && modelStatus.ready && (inferenceMode != CosyVoiceInferenceMode.INSTRUCT2 || instruction.isNotBlank()),
                        modifier = Modifier.fillMaxWidth()) { Text("合成并试听") }
                }
            }

            if (busyMessage.isNotBlank()) { Row(verticalAlignment = Alignment.CenterVertically) { CircularProgressIndicator(Modifier.padding(8.dp)); Text(busyMessage, modifier = Modifier.weight(1f)) } }
            if (resultMessage.isNotBlank()) { Card(Modifier.fillMaxWidth()) { Text(resultMessage, Modifier.padding(14.dp), fontSize = 14.sp) } }
            Spacer(Modifier.height(16.dp))
        }
    }
}

@Composable
private fun ModeButtons(selected: Int, enabled: Boolean, onSelected: (Int) -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        listOf(4 to "自动", 68 to "Buffer", 132 to "Image").forEach { (mode, label) -> ChoiceButton(label, selected == mode, enabled) { onSelected(mode) } }
    }
}

@Composable
private fun ChoiceButton(label: String, selected: Boolean, enabled: Boolean, onClick: () -> Unit) {
    if (selected) Button(onClick = onClick, enabled = enabled) { Text(label) } else OutlinedButton(onClick = onClick, enabled = enabled) { Text(label) }
}

private fun formatBytes(bytes: Long): String {
    val gib = bytes / 1024.0 / 1024.0 / 1024.0
    return if (gib >= 1.0) "%.2f GiB".format(gib) else "%.1f MiB".format(bytes / 1024.0 / 1024.0)
}
