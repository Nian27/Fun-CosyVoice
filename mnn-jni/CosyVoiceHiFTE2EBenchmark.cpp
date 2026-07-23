#include <MNN/ErrorCode.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kChannels = 80;
constexpr int kHarmonics = 9;
constexpr int kSampleRate = 24000;
constexpr int kMelHop = 480;
constexpr int kFftSize = 16;
constexpr int kStftHop = 4;
constexpr int kFftBins = 9;
constexpr double kPi = 3.14159265358979323846;

struct Stats {
    bool finite = true;
    float min = std::numeric_limits<float>::infinity();
    float max = -std::numeric_limits<float>::infinity();
    double mean = 0.0;
    double rms = 0.0;
};

struct ErrorStats {
    double mae = 0.0;
    double maxAbs = 0.0;
    double rms = 0.0;
};

struct ModelSession {
    std::shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session* session = nullptr;
};

std::string joinPath(const std::string& directory, const std::string& file) {
    if (directory.empty()) {
        return file;
    }
    const char last = directory.back();
    if (last == '/' || last == '\\') {
        return directory + file;
    }
#ifdef _WIN32
    return directory + "\\" + file;
#else
    return directory + "/" + file;
#endif
}

bool readFloats(const std::string& path, size_t count, std::vector<float>& values,
                bool quiet = false) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        if (!quiet) {
            std::cerr << "Cannot open " << path << std::endl;
        }
        return false;
    }
    const size_t expected = count * sizeof(float);
    if (static_cast<size_t>(input.tellg()) != expected) {
        if (!quiet) {
            std::cerr << "Unexpected size for " << path << ": expected " << expected
                      << " bytes, got " << input.tellg() << std::endl;
        }
        return false;
    }
    input.seekg(0);
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expected));
    return static_cast<bool>(input);
}

std::vector<float> generateNoiseTable(size_t count) {
    std::vector<float> values(count);
    uint32_t state = 1986u;
    for (size_t i = 0; i < count; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        values[i] = static_cast<float>(state >> 8) / 16777216.0f;
    }
    return values;
}

bool writeFloats(const std::string& path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        std::cerr << "Cannot create " << path << std::endl;
        return false;
    }
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    return static_cast<bool>(output);
}

void writeLittleEndian16(std::ofstream& output, uint16_t value) {
    const uint8_t bytes[2] = {static_cast<uint8_t>(value & 0xff),
                              static_cast<uint8_t>((value >> 8) & 0xff)};
    output.write(reinterpret_cast<const char*>(bytes), 2);
}

void writeLittleEndian32(std::ofstream& output, uint32_t value) {
    const uint8_t bytes[4] = {
        static_cast<uint8_t>(value & 0xff), static_cast<uint8_t>((value >> 8) & 0xff),
        static_cast<uint8_t>((value >> 16) & 0xff),
        static_cast<uint8_t>((value >> 24) & 0xff)};
    output.write(reinterpret_cast<const char*>(bytes), 4);
}

bool writePcm16Wave(const std::string& path, const std::vector<float>& pcm) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        std::cerr << "Cannot create " << path << std::endl;
        return false;
    }
    const uint32_t dataBytes = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
    output.write("RIFF", 4);
    writeLittleEndian32(output, 36 + dataBytes);
    output.write("WAVEfmt ", 8);
    writeLittleEndian32(output, 16);
    writeLittleEndian16(output, 1);
    writeLittleEndian16(output, 1);
    writeLittleEndian32(output, kSampleRate);
    writeLittleEndian32(output, kSampleRate * sizeof(int16_t));
    writeLittleEndian16(output, sizeof(int16_t));
    writeLittleEndian16(output, 16);
    output.write("data", 4);
    writeLittleEndian32(output, dataBytes);
    for (float sample : pcm) {
        const float clipped = std::max(-1.0f, std::min(1.0f, sample));
        const int16_t value = static_cast<int16_t>(std::lrint(clipped * 32767.0f));
        writeLittleEndian16(output, static_cast<uint16_t>(value));
    }
    return static_cast<bool>(output);
}

Stats calculateStats(const std::vector<float>& values) {
    Stats stats;
    long double sum = 0.0;
    long double squares = 0.0;
    for (float value : values) {
        if (!std::isfinite(value)) {
            stats.finite = false;
            continue;
        }
        stats.min = std::min(stats.min, value);
        stats.max = std::max(stats.max, value);
        sum += value;
        squares += static_cast<long double>(value) * value;
    }
    if (values.empty()) {
        stats.finite = false;
    } else {
        stats.mean = static_cast<double>(sum / values.size());
        stats.rms = std::sqrt(static_cast<double>(squares / values.size()));
    }
    return stats;
}

ErrorStats calculateError(const std::vector<float>& reference,
                          const std::vector<float>& candidate) {
    ErrorStats result;
    if (reference.size() != candidate.size() || reference.empty()) {
        result.maxAbs = std::numeric_limits<double>::infinity();
        return result;
    }
    long double absoluteSum = 0.0;
    long double squareSum = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double error = static_cast<double>(candidate[i]) - reference[i];
        const double absolute = std::abs(error);
        absoluteSum += absolute;
        squareSum += error * error;
        result.maxAbs = std::max(result.maxAbs, absolute);
    }
    result.mae = static_cast<double>(absoluteSum / reference.size());
    result.rms = std::sqrt(static_cast<double>(squareSum / reference.size()));
    return result;
}

bool copyToTensor(MNN::Tensor* tensor, const std::vector<float>& values) {
    if (static_cast<size_t>(tensor->elementSize()) != values.size()) {
        std::cerr << "Tensor element mismatch" << std::endl;
        return false;
    }
    MNN::Tensor host(tensor, MNN::Tensor::CAFFE);
    std::copy(values.begin(), values.end(), host.host<float>());
    tensor->copyFromHostTensor(&host);
    return true;
}

std::vector<float> copyFromTensor(MNN::Tensor* tensor) {
    MNN::Tensor host(tensor, MNN::Tensor::CAFFE);
    tensor->copyToHostTensor(&host);
    return std::vector<float>(host.host<float>(), host.host<float>() + host.elementSize());
}

ModelSession createModelSession(
    const std::string& path,
    MNNForwardType backend,
    MNN::BackendConfig::PrecisionMode precision,
    int threadsOrGpuMode,
    const std::string& cachePath = "") {
    ModelSession result;
    result.interpreter.reset(MNN::Interpreter::createFromFile(path.c_str()),
                             MNN::Interpreter::destroy);
    if (!result.interpreter) {
        return result;
    }
    if (!cachePath.empty()) {
        result.interpreter->setCacheFile(cachePath.c_str());
    }
    MNN::BackendConfig backendConfig;
    backendConfig.precision = precision;
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;
    backendConfig.power = MNN::BackendConfig::Power_High;
    MNN::ScheduleConfig schedule;
    schedule.type = backend;
    schedule.backupType = backend;
    schedule.numThread = threadsOrGpuMode;
    schedule.backendConfig = &backendConfig;
    result.session = result.interpreter->createSession(schedule);
    return result;
}

ModelSession createCpuSession(const std::string& path, int threads) {
    return createModelSession(
        path,
        MNN_FORWARD_CPU,
        MNN::BackendConfig::Precision_High,
        threads);
}

double elapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

void printStage(const std::string& stage) {
    std::ifstream status("/proc/self/status");
    std::string line;
    std::string rss;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            rss = line.substr(6);
            break;
        }
    }
    std::cout << "stage=" << stage;
    if (!rss.empty()) {
        std::cout << " vm_rss=" << rss;
    }
    std::cout << std::endl;
}

std::vector<float> generateSource(const std::vector<float>& f0,
                                  const std::vector<float>& noiseTable,
                                  const std::vector<float>& linearWeight, float linearBias) {
    const size_t melFrames = f0.size();
    const size_t sampleCount = melFrames * kMelHop;
    std::vector<float> source(sampleCount);
    double phase[kHarmonics] = {};
    for (size_t frame = 0; frame < melFrames; ++frame) {
        const bool voiced = f0[frame] > 10.0f;
        const float noiseAmplitude = voiced ? 0.003f : (0.1f / 3.0f);
        for (int harmonic = 0; harmonic < kHarmonics; ++harmonic) {
            phase[harmonic] += static_cast<double>(f0[frame]) * (harmonic + 1) /
                               kSampleRate * (2.0 * kPi * kMelHop);
        }
        for (int offset = 0; offset < kMelHop; ++offset) {
            const size_t sample = frame * kMelHop + offset;
            double merged = linearBias;
            for (int harmonic = 0; harmonic < kHarmonics; ++harmonic) {
                const float sine = 0.1f * static_cast<float>(std::sin(phase[harmonic]));
                const float noise = noiseAmplitude *
                                    noiseTable[sample * kHarmonics + harmonic];
                const float excitation = (voiced ? sine : 0.0f) + noise;
                merged += linearWeight[harmonic] * excitation;
            }
            source[sample] = static_cast<float>(std::tanh(merged));
        }
    }
    return source;
}

int reflectIndex(int index, int length) {
    if (index < 0) {
        return -index;
    }
    if (index >= length) {
        return 2 * length - 2 - index;
    }
    return index;
}

std::vector<float> stft(const std::vector<float>& source) {
    const size_t frames = source.size() / kStftHop + 1;
    std::vector<float> output(2 * kFftBins * frames);
    float window[kFftSize];
    for (int n = 0; n < kFftSize; ++n) {
        window[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * n / kFftSize));
    }
    for (size_t frame = 0; frame < frames; ++frame) {
        const int start = static_cast<int>(frame * kStftHop) - kFftSize / 2;
        for (int bin = 0; bin < kFftBins; ++bin) {
            double real = 0.0;
            double imag = 0.0;
            for (int n = 0; n < kFftSize; ++n) {
                const float sample = source[reflectIndex(start + n,
                                                         static_cast<int>(source.size()))] *
                                     window[n];
                const double angle = 2.0 * kPi * bin * n / kFftSize;
                real += sample * std::cos(angle);
                imag -= sample * std::sin(angle);
            }
            output[static_cast<size_t>(bin) * frames + frame] = static_cast<float>(real);
            output[static_cast<size_t>(kFftBins + bin) * frames + frame] =
                static_cast<float>(imag);
        }
    }
    return output;
}

std::vector<float> istft(const std::vector<float>& spectral, size_t frames) {
    const size_t paddedLength = (frames - 1) * kStftHop + kFftSize;
    std::vector<double> accumulated(paddedLength, 0.0);
    std::vector<double> envelope(paddedLength, 0.0);
    float window[kFftSize];
    for (int n = 0; n < kFftSize; ++n) {
        window[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * n / kFftSize));
    }
    for (size_t frame = 0; frame < frames; ++frame) {
        double real[kFftBins];
        double imag[kFftBins];
        for (int bin = 0; bin < kFftBins; ++bin) {
            const float logMagnitude = spectral[static_cast<size_t>(bin) * frames + frame];
            const float phaseRaw =
                spectral[static_cast<size_t>(kFftBins + bin) * frames + frame];
            const double magnitude = std::min(100.0, std::exp(static_cast<double>(logMagnitude)));
            const double phase = std::sin(static_cast<double>(phaseRaw));
            real[bin] = magnitude * std::cos(phase);
            imag[bin] = magnitude * std::sin(phase);
        }
        for (int n = 0; n < kFftSize; ++n) {
            double value = real[0] + real[kFftBins - 1] * (n % 2 == 0 ? 1.0 : -1.0);
            for (int bin = 1; bin < kFftBins - 1; ++bin) {
                const double angle = 2.0 * kPi * bin * n / kFftSize;
                value += 2.0 * (real[bin] * std::cos(angle) - imag[bin] * std::sin(angle));
            }
            value /= kFftSize;
            const size_t index = frame * kStftHop + n;
            accumulated[index] += value * window[n];
            envelope[index] += static_cast<double>(window[n]) * window[n];
        }
    }
    for (size_t i = 0; i < paddedLength; ++i) {
        if (envelope[i] > 1e-11) {
            accumulated[i] /= envelope[i];
        }
    }
    const size_t sampleCount = (frames - 1) * kStftHop;
    std::vector<float> pcm(sampleCount);
    for (size_t i = 0; i < sampleCount; ++i) {
        const double value = accumulated[i + kFftSize / 2];
        pcm[i] = static_cast<float>(std::max(-0.99, std::min(0.99, value)));
    }
    return pcm;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " hift-f0.mnn hift-core.mnn input_dir output_dir mel_frames threads"
                  << std::endl;
        return 2;
    }
    const std::string f0ModelPath = argv[1];
    const std::string coreModelPath = argv[2];
    const std::string inputDirectory = argv[3];
    const std::string outputDirectory = argv[4];
    const int melFrames = std::max(1, std::atoi(argv[5]));
    const int threads = std::max(1, std::atoi(argv[6]));
    const size_t sourceSamples = static_cast<size_t>(melFrames) * kMelHop;
    const size_t sourceFrames = static_cast<size_t>(melFrames) * 120 + 1;

    std::vector<float> mel;
    std::vector<float> noiseTable;
    std::vector<float> linearWeight;
    std::vector<float> linearBias;
    if (!readFloats(joinPath(inputDirectory, "mel.bin"),
                    static_cast<size_t>(kChannels) * melFrames, mel) ||
        !readFloats(joinPath(inputDirectory, "source-linear-weight.bin"), kHarmonics,
                    linearWeight) ||
        !readFloats(joinPath(inputDirectory, "source-linear-bias.bin"), 1, linearBias)) {
        return 3;
    }
    const bool externalNoise = readFloats(
        joinPath(inputDirectory, "source-noise-table.bin"),
        sourceSamples * kHarmonics, noiseTable, true);
    if (!externalNoise) {
        noiseTable = generateNoiseTable(sourceSamples * kHarmonics);
        std::cout << "source_noise=generated_xorshift32" << std::endl;
    }

    const auto allStart = std::chrono::steady_clock::now();
    printStage("f0_load_begin");
    ModelSession f0Model = createCpuSession(f0ModelPath, threads);
    if (!f0Model.session) {
        std::cerr << "Failed to create F0 session" << std::endl;
        return 4;
    }
    MNN::Tensor* f0Input = f0Model.interpreter->getSessionInput(f0Model.session, "mel");
    f0Model.interpreter->resizeTensor(f0Input, {1, kChannels, melFrames});
    f0Model.interpreter->resizeSession(f0Model.session);
    printStage("f0_run_begin");
    const auto f0Start = std::chrono::steady_clock::now();
    if (!copyToTensor(f0Input, mel) ||
        f0Model.interpreter->runSession(f0Model.session) != MNN::NO_ERROR) {
        std::cerr << "F0 inference failed" << std::endl;
        return 5;
    }
    std::vector<float> f0 = copyFromTensor(
        f0Model.interpreter->getSessionOutput(f0Model.session, "f0"));
    const double f0Ms = elapsedMs(f0Start);
    printStage("f0_run_done");

    const auto sourceStart = std::chrono::steady_clock::now();
    std::vector<float> source = generateSource(f0, noiseTable, linearWeight, linearBias[0]);
    std::vector<float> sourceStft = stft(source);
    const double sourceMs = elapsedMs(sourceStart);
    printStage("source_stft_done");

    printStage("core_load_begin");
    ModelSession coreModel = createCpuSession(coreModelPath, threads);
    if (!coreModel.session) {
        std::cerr << "Failed to create HiFT core session" << std::endl;
        return 6;
    }
    MNN::Tensor* coreMel = coreModel.interpreter->getSessionInput(coreModel.session, "mel");
    MNN::Tensor* coreSource =
        coreModel.interpreter->getSessionInput(coreModel.session, "source_stft");
    coreModel.interpreter->resizeTensor(coreMel, {1, kChannels, melFrames});
    coreModel.interpreter->resizeTensor(coreSource, {1, 2 * kFftBins,
                                                     static_cast<int>(sourceFrames)});
    coreModel.interpreter->resizeSession(coreModel.session);
    printStage("core_run_begin");
    const auto coreStart = std::chrono::steady_clock::now();
    if (!copyToTensor(coreMel, mel) || !copyToTensor(coreSource, sourceStft) ||
        coreModel.interpreter->runSession(coreModel.session) != MNN::NO_ERROR) {
        std::cerr << "HiFT core inference failed" << std::endl;
        return 7;
    }
    std::vector<float> spectral = copyFromTensor(
        coreModel.interpreter->getSessionOutput(coreModel.session, "spectral"));
    const double coreMs = elapsedMs(coreStart);
    printStage("core_run_done");

    const auto istftStart = std::chrono::steady_clock::now();
    std::vector<float> pcm = istft(spectral, sourceFrames);
    const double istftMs = elapsedMs(istftStart);
    const double totalMs = elapsedMs(allStart);
    printStage("istft_done");

    std::vector<float> referenceF0;
    std::vector<float> referenceSource;
    std::vector<float> referenceSourceStft;
    std::vector<float> referenceSpectral;
    std::vector<float> referencePcm;
    const bool referenceAvailable =
        readFloats(joinPath(inputDirectory, "f0.bin"), melFrames, referenceF0, true) &&
        readFloats(joinPath(inputDirectory, "source.bin"), sourceSamples,
                   referenceSource, true) &&
        readFloats(joinPath(inputDirectory, "source-stft.bin"),
                   2 * kFftBins * sourceFrames, referenceSourceStft, true) &&
        readFloats(joinPath(inputDirectory, "spectral.bin"),
                   2 * kFftBins * sourceFrames, referenceSpectral, true) &&
        readFloats(joinPath(inputDirectory, "pcm.bin"), sourceSamples, referencePcm, true);
    const ErrorStats f0Error = referenceAvailable ? calculateError(referenceF0, f0) : ErrorStats{};
    const ErrorStats sourceError = referenceAvailable
                                       ? calculateError(referenceSource, source)
                                       : ErrorStats{};
    const ErrorStats sourceStftError = referenceAvailable
                                           ? calculateError(referenceSourceStft, sourceStft)
                                           : ErrorStats{};
    const ErrorStats spectralError = referenceAvailable
                                         ? calculateError(referenceSpectral, spectral)
                                         : ErrorStats{};
    const ErrorStats pcmError = referenceAvailable ? calculateError(referencePcm, pcm) : ErrorStats{};
    const std::vector<float> referenceSpectralPcm =
        referenceAvailable ? istft(referenceSpectral, sourceFrames) : std::vector<float>{};
    const ErrorStats istftError = referenceAvailable
                                      ? calculateError(referencePcm, referenceSpectralPcm)
                                      : ErrorStats{};
    const Stats pcmStats = calculateStats(pcm);

    if (!writeFloats(joinPath(outputDirectory, "f0-android.bin"), f0) ||
        !writeFloats(joinPath(outputDirectory, "source-android.bin"), source) ||
        !writeFloats(joinPath(outputDirectory, "source-stft-android.bin"), sourceStft) ||
        !writeFloats(joinPath(outputDirectory, "spectral-android.bin"), spectral) ||
        !writeFloats(joinPath(outputDirectory, "pcm-android.bin"), pcm) ||
        !writePcm16Wave(joinPath(outputDirectory, "hift-android.wav"), pcm)) {
        return 9;
    }

    std::ofstream metrics(joinPath(outputDirectory, "hift_e2e_android.json"));
    metrics << std::fixed << std::setprecision(6)
            << "{\n"
            << "  \"melFrames\": " << melFrames << ",\n"
            << "  \"audioSeconds\": " << static_cast<double>(sourceSamples) / kSampleRate
            << ",\n"
            << "  \"f0Ms\": " << f0Ms << ",\n"
            << "  \"sourceStftMs\": " << sourceMs << ",\n"
            << "  \"coreMs\": " << coreMs << ",\n"
            << "  \"istftMs\": " << istftMs << ",\n"
            << "  \"totalMs\": " << totalMs << ",\n"
            << "  \"rtf\": " << totalMs / (static_cast<double>(sourceSamples) / kSampleRate * 1000.0)
            << ",\n"
            << "  \"referenceAvailable\": " << (referenceAvailable ? "true" : "false") << ",\n"
            << "  \"externalNoiseTable\": " << (externalNoise ? "true" : "false") << ",\n"
            << "  \"finite\": " << (pcmStats.finite ? "true" : "false") << ",\n"
            << "  \"pcmPeak\": " << std::max(std::abs(pcmStats.min), std::abs(pcmStats.max))
            << ",\n"
            << "  \"pcmRms\": " << pcmStats.rms << ",\n"
            << "  \"f0MaxAbsError\": " << f0Error.maxAbs << ",\n"
            << "  \"sourceMaxAbsError\": " << sourceError.maxAbs << ",\n"
            << "  \"sourceStftMaxAbsError\": " << sourceStftError.maxAbs << ",\n"
            << "  \"spectralMaxAbsError\": " << spectralError.maxAbs << ",\n"
            << "  \"pcmMaxAbsError\": " << pcmError.maxAbs << ",\n"
            << "  \"pcmMae\": " << pcmError.mae << ",\n"
            << "  \"istftReferenceMaxAbsError\": " << istftError.maxAbs << "\n"
            << "}\n";

    std::cout << std::fixed << std::setprecision(6)
              << "f0_ms=" << f0Ms << " source_stft_ms=" << sourceMs
              << " core_ms=" << coreMs << " istft_ms=" << istftMs
              << " total_ms=" << totalMs
              << " rtf=" << totalMs /
                                  (static_cast<double>(sourceSamples) / kSampleRate * 1000.0)
              << " f0_max_abs_error=" << f0Error.maxAbs
              << " reference_available=" << referenceAvailable
              << " source_max_abs_error=" << sourceError.maxAbs
              << " source_stft_max_abs_error=" << sourceStftError.maxAbs
              << " spectral_max_abs_error=" << spectralError.maxAbs
              << " pcm_max_abs_error=" << pcmError.maxAbs
              << " istft_reference_max_abs_error=" << istftError.maxAbs
              << " pcm_rms=" << pcmStats.rms
              << std::endl;
    const double pcmPeak = std::max(std::abs(pcmStats.min), std::abs(pcmStats.max));
    const bool referencePassed = !referenceAvailable ||
                                 (f0Error.maxAbs < 0.05 && sourceError.maxAbs < 0.001 &&
                                  sourceStftError.maxAbs < 0.005 && istftError.maxAbs < 0.001);
    return pcmStats.finite && referencePassed &&
                   pcmPeak <= 0.99 && pcmStats.rms > 0.001 && pcmStats.rms < 0.5
               ? 0
               : 10;
}
