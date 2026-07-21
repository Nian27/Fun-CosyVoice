package com.fun.cosyvoice

internal object CosyVoiceLlmNative {
    init {
        System.loadLibrary("cosy_llm_jni")
    }

    external fun run(
        configPath: String,
        promptsPath: String,
        maxTokens: Int,
        promptSpeechTokensPath: String,
        outputDirectory: String,
        appendPromptSpeechTokens: Boolean
    ): Int

    external fun reset()
}
