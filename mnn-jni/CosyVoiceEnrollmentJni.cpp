#include <jni.h>
#include <android/log.h>

#include <MNN/ErrorCode.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/Module.hpp>
#include <audio/audio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr const char* kLogTag = "CosyVoiceEnrollment";

void logError(const char* stage, int code) {
    __android_log_print(ANDROID_LOG_ERROR, kLogTag, "%s failed code=%d", stage, code);
}

using MNN::Express::NHWC;
using MNN::Express::REFLECT;
using MNN::Express::VARP;
using MNN::Express::_Const;
using MNN::Express::_Log;
using MNN::Express::_Maximum;
using MNN::Express::_Pad;
using MNN::Express::_Scalar;

constexpr float kPi = 3.14159265358979323846f;

class UtfChars {
public:
    UtfChars(JNIEnv* env, jstring value)
        : env_(env), value_(value), chars_(value ? env->GetStringUTFChars(value, nullptr) : nullptr) {}
    ~UtfChars() {
        if (chars_) env_->ReleaseStringUTFChars(value_, chars_);
    }
    const char* get() const { return chars_ ? chars_ : ""; }

private:
    JNIEnv* env_;
    jstring value_;
    const char* chars_;
};

std::string joinPath(const std::string& directory, const std::string& name) {
    if (!directory.empty() && directory.back() != '/' && directory.back() != '\\') {
        return directory + "/" + name;
    }
    return directory + name;
}

std::string parentDirectory(const std::string& path) {
    const auto position = path.find_last_of("/\\");
    return position == std::string::npos ? "." : path.substr(0, position);
}

double elapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

bool finiteVector(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

bool readFloats(const std::string& path, size_t count, std::vector<float>& values) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) != count * sizeof(float)) return false;
    input.seekg(0);
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(input);
}

bool writeFloats(const std::string& path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    return static_cast<bool>(output);
}

std::vector<float> sincResample(const float* input, size_t inputCount, int inputRate, int outputRate) {
    if (inputRate == outputRate) return std::vector<float>(input, input + inputCount);
    const double ratio = static_cast<double>(outputRate) / inputRate;
    const size_t outputCount = static_cast<size_t>(std::floor(inputCount * ratio));
    const int radius = 24;
    const double cutoff = std::min(1.0, ratio) * 0.94;
    std::vector<float> output(outputCount);
    for (size_t i = 0; i < outputCount; ++i) {
        const double position = static_cast<double>(i) / ratio;
        const int center = static_cast<int>(std::floor(position));
        double sum = 0.0;
        double weightSum = 0.0;
        for (int sample = center - radius + 1; sample <= center + radius; ++sample) {
            if (sample < 0 || sample >= static_cast<int>(inputCount)) continue;
            const double distance = position - sample;
            const double normalized = distance / radius;
            if (std::abs(normalized) >= 1.0) continue;
            const double sincArg = kPi * cutoff * distance;
            const double sinc = std::abs(sincArg) < 1e-9 ? 1.0 : std::sin(sincArg) / sincArg;
            const double window = 0.5 + 0.5 * std::cos(kPi * normalized);
            const double weight = cutoff * sinc * window;
            sum += input[sample] * weight;
            weightSum += weight;
        }
        output[i] = static_cast<float>(weightSum == 0.0 ? 0.0 : sum / weightSum);
    }
    return output;
}

MNN::ScheduleConfig cpuSchedule(int threads, MNN::BackendConfig& backend) {
    backend.precision = MNN::BackendConfig::Precision_High;
    backend.memory = MNN::BackendConfig::Memory_Normal;
    backend.power = MNN::BackendConfig::Power_High;
    MNN::ScheduleConfig schedule;
    schedule.type = MNN_FORWARD_CPU;
    schedule.backupType = MNN_FORWARD_CPU;
    schedule.numThread = threads;
    schedule.backendConfig = &backend;
    return schedule;
}

int runSpeechTokenizer(const std::string& modelPath, const std::vector<float>& features,
                       int frames, int threads, std::vector<int32_t>& tokens, double& inferenceMs) {
    MNN::BackendConfig backend;
    auto schedule = cpuSchedule(threads, backend);
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backend, threads);
    MNN::Express::ExecutorScope executorScope(executor);
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime(
        MNN::Express::Executor::RuntimeManager::createRuntimeManager(schedule));
    if (!runtime) {
        logError("speech-tokenizer/create-runtime", 20);
        return 20;
    }
    runtime->setExternalPath(parentDirectory(modelPath), 3);
    // Dynamic Module execution may materialize subgraphs only on the first
    // forward. Keep the external weight file bound for that whole lifetime;
    // Module::load's temporary default binding is cleared too early.
    if (std::ifstream(modelPath + ".weight").good()) {
        runtime->setExternalFile(modelPath + ".weight");
    }
    MNN::Express::Module::Config moduleConfig;
    moduleConfig.shapeMutable = true;
    moduleConfig.dynamic = true;
    const std::vector<std::string> inputNames = {"feats", "feats_length"};
    const std::vector<std::string> outputNames = {"indices"};
    std::shared_ptr<MNN::Express::Module> module(
        MNN::Express::Module::load(inputNames, outputNames, modelPath.c_str(), runtime, &moduleConfig));
    if (!module) {
        logError("speech-tokenizer/load-module", 21);
        return 21;
    }
    const auto moduleInfo = module->getInfo();
    if (!moduleInfo || moduleInfo->inputs.size() != 2) return 22;
    auto featureInput = MNN::Express::_Input(
        {1, 128, frames}, moduleInfo->inputs[0].order, moduleInfo->inputs[0].type);
    auto lengthInput = MNN::Express::_Input(
        {1}, moduleInfo->inputs[1].order, moduleInfo->inputs[1].type);
    const auto featureInfo = featureInput->getInfo();
    if (!featureInfo || featureInfo->size != features.size() ||
        featureInfo->type != halide_type_of<float>()) return 24;
    std::copy(features.begin(), features.end(), featureInput->writeMap<float>());
    const auto lengthInfo = lengthInput->getInfo();
    if (!lengthInfo || lengthInfo->size != 1 ||
        lengthInfo->type != halide_type_of<int32_t>()) return 22;
    lengthInput->writeMap<int32_t>()[0] = frames;
    const auto start = std::chrono::steady_clock::now();
    auto outputs = module->onForward({featureInput, lengthInput});
    if (outputs.size() != 1) return 25;
    const auto outputInfo = outputs[0]->getInfo();
    const int32_t* outputData = outputs[0]->readMap<int32_t>();
    inferenceMs = elapsedMs(start);
    if (!outputInfo || !outputData || outputInfo->type != halide_type_of<int32_t>()) return 26;
    tokens.assign(outputData, outputData + outputInfo->size);
    if (tokens.empty()) return 27;
    const auto minMax = std::minmax_element(tokens.begin(), tokens.end());
    const std::set<int32_t> unique(tokens.begin(), tokens.end());
    if (*minMax.first < 0 || *minMax.second > 6560 || unique.size() < 4) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag,
                            "speech-tokenizer invalid count=%zu min=%d max=%d unique=%zu",
                            tokens.size(), *minMax.first, *minMax.second, unique.size());
        return 28;
    }
    return 0;
}

int runCampPlus(const std::string& modelPath, const std::vector<float>& features,
                int frames, int threads, std::vector<float>& embedding, double& inferenceMs) {
    std::shared_ptr<MNN::Interpreter> interpreter(
        MNN::Interpreter::createFromFile(modelPath.c_str()), MNN::Interpreter::destroy);
    if (!interpreter) {
        logError("campplus/create-interpreter", 30);
        return 30;
    }
    interpreter->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
    MNN::BackendConfig backend;
    auto schedule = cpuSchedule(threads, backend);
    MNN::Session* session = interpreter->createSession(schedule);
    if (!session) {
        logError("campplus/create-session", 31);
        return 31;
    }
    MNN::Tensor* input = interpreter->getSessionInput(session, "input");
    if (!input) return 32;
    interpreter->resizeTensor(input, {1, frames, 80});
    interpreter->resizeSession(session);
    MNN::Tensor hostInput(input, MNN::Tensor::CAFFE);
    if (hostInput.elementSize() != features.size()) return 34;
    std::copy(features.begin(), features.end(), hostInput.host<float>());
    input->copyFromHostTensor(&hostInput);
    const auto start = std::chrono::steady_clock::now();
    if (interpreter->runSession(session) != MNN::NO_ERROR) return 35;
    inferenceMs = elapsedMs(start);
    MNN::Tensor* output = interpreter->getSessionOutput(session, "output");
    if (!output) return 36;
    MNN::Tensor hostOutput(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(&hostOutput);
    embedding.assign(hostOutput.host<float>(), hostOutput.host<float>() + hostOutput.elementSize());
    return embedding.size() == 192 && finiteVector(embedding) ? 0 : 37;
}

std::vector<float> extractWhisperFeatures(const std::vector<float>& audio, int& frames) {
    auto waveform = _Const(audio.data(), {static_cast<int>(audio.size())}, NHWC, halide_type_of<float>());
    auto features = MNN::AUDIO::whisper_fbank(waveform, 16000, 128, 400, 160, 0);
    auto info = features->getInfo();
    if (!info || info->size == 0) return {};
    frames = info->dim.back();
    const float* data = features->readMap<float>();
    return std::vector<float>(data, data + info->size);
}

std::vector<float> extractCampFeatures(const std::vector<float>& audio, int& frames) {
    auto waveform = _Const(audio.data(), {static_cast<int>(audio.size())}, NHWC, halide_type_of<float>());
    auto features = MNN::AUDIO::fbank(waveform, 16000, 80, 400, 160, 0.0f, 0.97f);
    if (features == nullptr || !features->getInfo()) return {};
    auto info = features->getInfo();
    frames = info->dim[0];
    const float* source = features->readMap<float>();
    std::vector<float> result(source, source + info->size);
    for (int bin = 0; bin < 80; ++bin) {
        double sum = 0.0;
        for (int frame = 0; frame < frames; ++frame) sum += result[frame * 80 + bin];
        const float mean = static_cast<float>(sum / frames);
        for (int frame = 0; frame < frames; ++frame) result[frame * 80 + bin] -= mean;
    }
    return result;
}

std::vector<float> extractPromptMel(const std::vector<float>& audio, int& frames) {
    auto waveform = _Const(audio.data(), {static_cast<int>(audio.size())}, NHWC, halide_type_of<float>());
    const int padding[] = {720, 720};
    waveform = _Pad(waveform, _Const(padding, {2}, NHWC, halide_type_of<int>()), REFLECT);
    MNN::AUDIO::MelscaleParams mel;
    mel.n_mels = 80;
    mel.n_fft = 1920;
    mel.sample_rate = 24000;
    mel.htk = false;
    mel.norm = true;
    mel.f_min = 0.0f;
    mel.f_max = 12000.0f;
    MNN::AUDIO::SpectrogramParams spectrum;
    spectrum.n_fft = 1920;
    spectrum.hop_length = 480;
    spectrum.win_length = 1920;
    spectrum.window_type = MNN::AUDIO::HANNING;
    spectrum.center = false;
    spectrum.normalized = false;
    spectrum.power = 1.0f;
    auto features = MNN::AUDIO::mel_spectrogram(waveform, &mel, &spectrum);
    features = _Log(_Maximum(features, _Scalar<float>(1e-5f)));
    auto info = features->getInfo();
    if (!info || info->dim.size() != 2 || info->dim[1] != 80) return {};
    frames = info->dim[0];
    const float* source = features->readMap<float>();
    std::vector<float> channelMajor(static_cast<size_t>(80) * frames);
    for (int frame = 0; frame < frames; ++frame) {
        for (int bin = 0; bin < 80; ++bin) {
            channelMajor[static_cast<size_t>(bin) * frames + frame] = source[frame * 80 + bin];
        }
    }
    return channelMajor;
}

bool projectSpeaker(const std::vector<float>& embedding, const std::string& weightPath,
                    const std::string& biasPath, std::vector<float>& speaker) {
    std::vector<float> weights;
    std::vector<float> bias;
    if (!readFloats(weightPath, 80 * 192, weights) || !readFloats(biasPath, 80, bias)) return false;
    double norm = 0.0;
    for (float value : embedding) norm += static_cast<double>(value) * value;
    norm = std::sqrt(std::max(norm, 1e-12));
    speaker.resize(80);
    for (int output = 0; output < 80; ++output) {
        double value = bias[output];
        for (int input = 0; input < 192; ++input) {
            value += weights[output * 192 + input] * (embedding[input] / norm);
        }
        speaker[output] = static_cast<float>(value);
    }
    return finiteVector(speaker);
}

int enroll(const std::string& tokenizerPath, const std::string& campPath,
           const std::string& affineWeightPath, const std::string& affineBiasPath,
           const std::string& wavPath, const std::string& outputDirectory, int threads) {
    const auto totalStart = std::chrono::steady_clock::now();
    auto loaded = MNN::AUDIO::load(wavPath, 0);
    if (loaded.first == nullptr || loaded.second < 16000) return 10;
    auto info = loaded.first->getInfo();
    if (!info || info->size < loaded.second * 3 || info->size > loaded.second * 15) return 11;
    const float* source = loaded.first->readMap<float>();
    std::vector<float> audio16 = sincResample(source, info->size, loaded.second, 16000);
    std::vector<float> audio24 = sincResample(source, info->size, loaded.second, 24000);

    int whisperFrames = 0;
    int campFrames = 0;
    int promptFrames = 0;
    auto whisperFeatures = extractWhisperFeatures(audio16, whisperFrames);
    auto campFeatures = extractCampFeatures(audio16, campFrames);
    auto promptMel = extractPromptMel(audio24, promptFrames);
    if (whisperFeatures.empty() || campFeatures.empty() || promptMel.empty()) return 12;

    std::vector<int32_t> tokens;
    std::vector<float> embedding;
    double tokenizerMs = 0.0;
    double campMs = 0.0;
    int code = runSpeechTokenizer(tokenizerPath, whisperFeatures, whisperFrames, threads, tokens, tokenizerMs);
    if (code != 0) return code;
    code = runCampPlus(campPath, campFeatures, campFrames, threads, embedding, campMs);
    if (code != 0) return code;
    // Keep the per-sentence zero-shot prompt bounded. CAMPPlus still sees the
    // complete reference clip above, while LLM/Flow only carry this prefix on
    // every synthesis request.
    constexpr int kMaxRealtimePromptTokens = 125;
    const int alignedTokens = std::min(
        {static_cast<int>(tokens.size()), promptFrames / 2, kMaxRealtimePromptTokens});
    if (alignedTokens <= 0) return 40;
    tokens.resize(alignedTokens);
    const int alignedFrames = alignedTokens * 2;
    if (alignedFrames != promptFrames) {
        std::vector<float> aligned(static_cast<size_t>(80) * alignedFrames);
        for (int channel = 0; channel < 80; ++channel) {
            std::copy(promptMel.begin() + static_cast<size_t>(channel) * promptFrames,
                      promptMel.begin() + static_cast<size_t>(channel) * promptFrames + alignedFrames,
                      aligned.begin() + static_cast<size_t>(channel) * alignedFrames);
        }
        promptMel.swap(aligned);
        promptFrames = alignedFrames;
    }
    std::vector<float> speaker;
    if (!projectSpeaker(embedding, affineWeightPath, affineBiasPath, speaker)) return 41;

    std::ofstream tokenOutput(joinPath(outputDirectory, "prompt-speech-tokens.csv"));
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) tokenOutput << ',';
        tokenOutput << tokens[i];
    }
    if (!tokenOutput || !writeFloats(joinPath(outputDirectory, "prompt-cond.bin"), promptMel) ||
        !writeFloats(joinPath(outputDirectory, "spks.bin"), speaker)) return 42;

    std::ofstream report(joinPath(outputDirectory, "enrollment-report.json"));
    report << std::fixed << std::setprecision(3)
           << "{\n"
           << "  \"inputSampleRate\": " << loaded.second << ",\n"
           << "  \"inputSeconds\": " << static_cast<double>(info->size) / loaded.second << ",\n"
           << "  \"whisperFrames\": " << whisperFrames << ",\n"
           << "  \"campFrames\": " << campFrames << ",\n"
           << "  \"promptTokens\": " << tokens.size() << ",\n"
           << "  \"promptFrames\": " << promptFrames << ",\n"
           << "  \"tokenizerMs\": " << tokenizerMs << ",\n"
           << "  \"campMs\": " << campMs << ",\n"
           << "  \"totalMs\": " << elapsedMs(totalStart) << "\n"
           << "}\n";
    return report ? 0 : 43;
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_io_legado_app_cosy_CosyVoiceEnrollmentNative_enroll(
    JNIEnv* env, jobject, jstring tokenizerPath, jstring campPath,
    jstring affineWeightPath, jstring affineBiasPath, jstring wavPath,
    jstring outputDirectory, jint threads) {
    UtfChars tokenizer(env, tokenizerPath);
    UtfChars camp(env, campPath);
    UtfChars weights(env, affineWeightPath);
    UtfChars bias(env, affineBiasPath);
    UtfChars wav(env, wavPath);
    UtfChars output(env, outputDirectory);
    if (env->ExceptionCheck()) return 1;
    return enroll(tokenizer.get(), camp.get(), weights.get(), bias.get(), wav.get(), output.get(),
                  std::max(1, static_cast<int>(threads)));
}
