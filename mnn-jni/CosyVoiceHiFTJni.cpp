#include <jni.h>

#include <array>
#include <mutex>
#include <string>

#define COSYVOICE_HIFT_LIBRARY
#include "CosyVoiceHiFTPersistentBenchmark.cpp"

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

std::mutex& hiftMutex() {
    static std::mutex mutex;
    return mutex;
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_io_legado_app_cosy_CosyVoiceHiFTNative_run(
    JNIEnv* env,
    jobject,
    jstring f0ModelPath,
    jstring coreModelPath,
    jstring manifestPath,
    jint threads,
    jstring reportPath,
    jstring coreBackend,
    jstring corePrecision,
    jstring coreCachePath,
    jint coreGpuMode) {
    UtfChars f0Model(env, f0ModelPath);
    UtfChars coreModel(env, coreModelPath);
    UtfChars manifest(env, manifestPath);
    UtfChars report(env, reportPath);
    UtfChars backend(env, coreBackend);
    UtfChars precision(env, corePrecision);
    UtfChars cache(env, coreCachePath);
    if (env->ExceptionCheck()) {
        return 12;
    }

    std::string threadCount = std::to_string(threads);
    std::string gpuMode = std::to_string(coreGpuMode);
    std::array<char*, 12> argv = {
        const_cast<char*>("cosy_hift_jni"),
        const_cast<char*>(f0Model.get()),
        const_cast<char*>(coreModel.get()),
        const_cast<char*>(manifest.get()),
        const_cast<char*>(threadCount.c_str()),
        const_cast<char*>(report.get()),
        const_cast<char*>(""),
        const_cast<char*>(""),
        const_cast<char*>(backend.get()),
        const_cast<char*>(precision.get()),
        const_cast<char*>(cache.get()),
        const_cast<char*>(gpuMode.c_str()),
    };
    std::lock_guard<std::mutex> lock(hiftMutex());
    return cosyVoiceHiFTMain(static_cast<int>(argv.size()), argv.data());
}

extern "C" JNIEXPORT void JNICALL
Java_io_legado_app_cosy_CosyVoiceHiFTNative_reset(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(hiftMutex());
    resetHiFTRuntime();
}
