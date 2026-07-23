#include <jni.h>

#include <array>
#include <string>

#define COSYVOICE_FLOW_LIBRARY
#include "CosyVoiceFlowPersistentBenchmark.cpp"

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

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_io_legado_app_cosy_CosyVoiceFlowNative_run(
    JNIEnv* env,
    jobject,
    jstring modelPath,
    jstring manifestPath,
    jstring backend,
    jstring precision,
    jint threads,
    jstring reportPath,
    jstring cachePath) {
    UtfChars model(env, modelPath);
    UtfChars manifest(env, manifestPath);
    UtfChars backendName(env, backend);
    UtfChars precisionName(env, precision);
    UtfChars report(env, reportPath);
    UtfChars cache(env, cachePath);
    if (env->ExceptionCheck()) {
        return 12;
    }

    std::string threadCount = std::to_string(threads);
    std::array<char*, 11> argv = {
        const_cast<char*>("cosy_flow_jni"),
        const_cast<char*>(model.get()),
        const_cast<char*>(manifest.get()),
        const_cast<char*>(backendName.get()),
        const_cast<char*>(precisionName.get()),
        const_cast<char*>(threadCount.c_str()),
        const_cast<char*>(report.get()),
        const_cast<char*>(""),
        const_cast<char*>(""),
        const_cast<char*>("0"),
        const_cast<char*>(cache.get()),
    };
    return cosyVoiceFlowMain(static_cast<int>(argv.size()), argv.data());
}

extern "C" JNIEXPORT void JNICALL
Java_io_legado_app_cosy_CosyVoiceFlowNative_reset(JNIEnv*, jobject) {
    resetFlowRuntime();
}
