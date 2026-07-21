package com.fun.cosyvoice

internal object CosyVoiceEnrollmentNative {
    init {
        System.loadLibrary("cosy_enrollment_jni")
    }

    external fun enroll(
        tokenizerModelPath: String,
        campPlusModelPath: String,
        affineWeightPath: String,
        affineBiasPath: String,
        sourceWavPath: String,
        outputDirectory: String,
        threads: Int
    ): Int

    fun errorMessage(code: Int): String = when (code) {
        10 -> "参考 WAV 无法读取"
        11 -> "参考人声长度不在 3-15 秒"
        12 -> "参考音频特征提取失败"
        20 -> "语音 Tokenizer 模型无法打开"
        21 -> "语音 Tokenizer Session 创建失败"
        22 -> "语音 Tokenizer 输入节点不匹配"
        24 -> "语音 Tokenizer 输入尺寸不匹配"
        25 -> "语音 Tokenizer 推理失败"
        26 -> "语音 Tokenizer 输出节点不匹配"
        27 -> "语音 Tokenizer 没有生成 Token"
        28 -> "语音 Tokenizer 输出无效，请重新导入新版音色创建扩展"
        30 -> "说话人模型无法打开"
        31 -> "说话人模型 Session 创建失败"
        32 -> "说话人模型输入节点不匹配"
        34 -> "说话人模型输入尺寸不匹配"
        35 -> "说话人特征推理失败"
        36 -> "说话人模型输出节点不匹配"
        37 -> "说话人特征无效"
        40 -> "参考音频生成的 Token 数量超出限制"
        41 -> "说话人特征投影失败"
        42 -> "音色档案文件写入失败"
        else -> "未知原生错误"
    }
}
