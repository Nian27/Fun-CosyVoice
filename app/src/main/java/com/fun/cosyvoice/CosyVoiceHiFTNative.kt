package com.fun.cosyvoice

internal object CosyVoiceHiFTNative {
    init {
        System.loadLibrary("cosy_hift_jni")
    }

    external fun run(
        f0ModelPath: String,
        coreModelPath: String,
        manifestPath: String,
        threads: Int,
        reportPath: String,
        coreBackend: String,
        corePrecision: String,
        coreCachePath: String,
        coreGpuMode: Int
    ): Int

    external fun reset()
}
