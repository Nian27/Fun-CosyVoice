package com.fun.cosyvoice

import android.content.Context
import android.os.StatFs
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.job
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.RandomAccessFile
import java.util.concurrent.TimeUnit

data class CosyVoiceModelDownloadProgress(
    val fileName: String,
    val fileIndex: Int,
    val fileCount: Int,
    val fileBytes: Long,
    val fileTotalBytes: Long,
    val totalBytes: Long,
    val totalExpectedBytes: Long
) {
    val percent: Int
        get() = if (totalExpectedBytes <= 0L) 0 else
            (totalBytes * 100L / totalExpectedBytes).toInt().coerceIn(0, 100)
}

class CosyVoiceModelDownloader(
    context: Context,
    private val store: CosyVoiceStore = CosyVoiceStore(context.applicationContext)
) {
    companion object {
        private const val TAG = "CosyVoiceDownloader"
        private const val REPOSITORY = "cstr/cosyvoice3-0.5b-2512-GGUF"
        private const val EXTRA_FREE_BYTES = 256L * 1024L * 1024L
    }

    private val appContext = context.applicationContext
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .callTimeout(0, TimeUnit.MILLISECONDS)
        .retryOnConnectionFailure(true)
        .build()

    suspend fun download(onProgress: (CosyVoiceModelDownloadProgress) -> Unit) {
        val specs = CosyVoiceStore.MODEL_FILE_SPECS
        val expectedTotal = specs.sumOf { it.bytes }
        val requiredBytes = specs.sumOf { spec ->
            val target = store.modelFile(spec)
            val marker = store.modelVerifiedFile(spec)
            if (target.length() == spec.bytes && marker.readTextOrEmpty() == spec.sha256) 0L else spec.bytes
        }
        val available = StatFs(appContext.filesDir.absolutePath).availableBytes
        check(available >= requiredBytes + EXTRA_FREE_BYTES) {
            "存储空间不足：还需 ${formatBytes(requiredBytes + EXTRA_FREE_BYTES)}，当前可用 ${formatBytes(available)}"
        }

        var completedBytes = 0L
        specs.forEachIndexed { index, spec ->
            currentCoroutineContext().ensureActive()
            val target = store.modelFile(spec)
            val marker = store.modelVerifiedFile(spec)
            if (target.length() == spec.bytes && marker.readTextOrEmpty() == spec.sha256) {
                completedBytes += spec.bytes
                onProgress(progress(spec, index, specs.size, spec.bytes, completedBytes, expectedTotal))
                return@forEachIndexed
            }
            if (target.length() == spec.bytes) {
                val valid = runCatching { store.verifyModelFile(target, spec, checkHash = true) }.isSuccess
                if (valid) {
                    marker.writeText(spec.sha256)
                    completedBytes += spec.bytes
                    onProgress(progress(spec, index, specs.size, spec.bytes, completedBytes, expectedTotal))
                    return@forEachIndexed
                }
                Log.w(TAG, "existing model checksum mismatch name=${spec.name}")
            }
            downloadFile(spec, index, specs.size, completedBytes, expectedTotal, onProgress)
            completedBytes += spec.bytes
        }
        Log.i(TAG, "online model download complete bytes=$expectedTotal")
    }

    private suspend fun downloadFile(
        spec: CosyVoiceModelFileSpec, fileIndex: Int, fileCount: Int,
        completedBytes: Long, expectedTotal: Long,
        onProgress: (CosyVoiceModelDownloadProgress) -> Unit
    ) {
        val target = store.modelFile(spec)
        val part = store.modelPartFile(spec)
        val marker = store.modelVerifiedFile(spec)
        if (part.length() > spec.bytes) part.delete()
        var offset = part.length()
        val url = "https://huggingface.co/$REPOSITORY/resolve/main/${spec.name}?download=true"
        val request = Request.Builder()
            .url(url)
            .header("Accept-Encoding", "identity")
            .apply { if (offset > 0L) header("Range", "bytes=$offset-") }
            .build()
        val call = client.newCall(request)
        val cancellationHandle = currentCoroutineContext().job.invokeOnCompletion { cause ->
            if (cause is CancellationException) call.cancel()
        }
        try {
            call.execute().use { response ->
                check(response.isSuccessful) { "下载 ${spec.name} 失败：HTTP ${response.code}" }
                val append = offset > 0L && response.code == 206
                if (!append) { offset = 0L; part.delete() }
                val body = response.body
                RandomAccessFile(part, "rw").use { output ->
                    output.seek(offset)
                    body.byteStream().use { input ->
                        val buffer = ByteArray(1024 * 1024)
                        var downloaded = offset
                        var lastProgressAt = 0L
                        while (true) {
                            currentCoroutineContext().ensureActive()
                            val count = input.read(buffer)
                            if (count < 0) break
                            output.write(buffer, 0, count)
                            downloaded += count
                            check(downloaded <= spec.bytes) { "${spec.name} 下载数据超过预期大小" }
                            val now = System.currentTimeMillis()
                            if (now - lastProgressAt >= 250L || downloaded == spec.bytes) {
                                onProgress(progress(spec, fileIndex, fileCount, downloaded,
                                    completedBytes + downloaded, expectedTotal))
                                lastProgressAt = now
                            }
                        }
                    }
                }
            }
            store.verifyModelFile(part, spec, checkHash = true)
            marker.delete()
            target.delete()
            check(part.renameTo(target)) { "${spec.name} 无法保存" }
            marker.writeText(spec.sha256)
            onProgress(progress(spec, fileIndex, fileCount, spec.bytes, completedBytes + spec.bytes, expectedTotal))
        } finally {
            cancellationHandle.dispose()
        }
    }

    private fun progress(spec: CosyVoiceModelFileSpec, index: Int, count: Int,
                         fileBytes: Long, totalBytes: Long, expectedTotal: Long) =
        CosyVoiceModelDownloadProgress(
            fileName = spec.name, fileIndex = index + 1, fileCount = count,
            fileBytes = fileBytes, fileTotalBytes = spec.bytes,
            totalBytes = totalBytes, totalExpectedBytes = expectedTotal
        )

    private fun File.readTextOrEmpty(): String =
        runCatching { if (exists()) readText().trim() else "" }.getOrDefault("")

    private fun formatBytes(bytes: Long): String = "%.1f MB".format(bytes / 1024.0 / 1024.0)
}
