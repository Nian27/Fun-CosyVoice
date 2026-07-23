#include <jni.h>

#include <array>
#include <mutex>
#include <string>

#define COSYVOICE_LLM_LIBRARY
#include "CosyVoiceLlmPersistentBenchmark.cpp"

namespace {

class UtfChars {
public:
    UtfChars(JNIEnv* env, jstring value)
        : env_(env), value_(value), chars_(value ? env->GetStringUTFChars(value, nullptr) : nullptr) {
    }

    ~UtfChars() {
        if (chars_) {
            env_->ReleaseStringUTFChars(value_, chars_);
        }
    }

    const char* get() const {
        return chars_ ? chars_ : "";
    }

private:
    JNIEnv* env_;
    jstring value_;
    const char* chars_;
};

std::mutex& llmMutex() {
    static std::mutex mutex;
    return mutex;
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_io_legado_app_cosy_CosyVoiceLlmNative_run(
    JNIEnv* env,
    jobject,
    jstring configPath,
    jstring promptsPath,
    jint maxTokens,
    jstring promptSpeechTokensPath,
    jstring outputDirectory,
    jboolean appendPromptSpeechTokens) {
    UtfChars config(env, configPath);
    UtfChars prompts(env, promptsPath);
    UtfChars promptTokens(env, promptSpeechTokensPath);
    UtfChars output(env, outputDirectory);
    if (env->ExceptionCheck()) {
        return 8;
    }

    std::string maxTokenCount = std::to_string(maxTokens);
    const char* appendPromptTokens = appendPromptSpeechTokens == JNI_TRUE ? "1" : "0";
    std::array<const char*, 9> argv = {
        "cosy_llm_jni",
        config.get(),
        prompts.get(),
        maxTokenCount.c_str(),
        promptTokens.get(),
        output.get(),
        "",
        "",
        appendPromptTokens,
    };
    std::lock_guard<std::mutex> lock(llmMutex());
    return cosyVoiceLlmMain(static_cast<int>(argv.size()), argv.data());
}

extern "C" JNIEXPORT void JNICALL
Java_io_legado_app_cosy_CosyVoiceLlmNative_reset(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(llmMutex());
    resetLlmRuntime();
}
