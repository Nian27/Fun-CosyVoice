package com.fun.cosyvoice

internal object CosyVoiceFlowNative {
    init {
        System.loadLibrary("cosy_flow_jni")
    }

    external fun run(
        modelPath: String,
        manifestPath: String,
        backend: String,
        precision: String,
        threads: Int,
        reportPath: String,
        cachePath: String
    ): Int

    external fun reset()
}
