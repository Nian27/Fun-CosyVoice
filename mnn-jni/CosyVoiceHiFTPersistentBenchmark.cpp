#include <sstream>
#include <thread>

#define main CosyVoiceHiFTStandaloneMain
#include "CosyVoiceHiFTE2EBenchmark.cpp"
#undef main

namespace {

struct HiFTRequest {
    std::string inputDirectory;
    std::string outputDirectory;
    int melFrames = 0;
};

std::string currentRss() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            return line.substr(6);
        }
    }
    return {};
}

void signalReadyAndWait(const std::string& readyPath, const std::string& startPath) {
    if (readyPath.empty() || startPath.empty()) {
        return;
    }
    std::ofstream(readyPath) << "ready\n";
    while (!std::ifstream(startPath).good()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

std::vector<HiFTRequest> readHiFTManifest(const std::string& path) {
    std::ifstream input(path);
    std::vector<HiFTRequest> requests;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::istringstream fields(line);
        HiFTRequest request;
        if (fields >> request.inputDirectory >> request.outputDirectory >> request.melFrames &&
            request.melFrames > 0) {
            requests.push_back(request);
        }
    }
    return requests;
}

MNNForwardType parseHiFTBackend(const std::string& value) {
    if (value == "cpu") {
        return MNN_FORWARD_CPU;
    }
    if (value == "opencl") {
        return MNN_FORWARD_OPENCL;
    }
    return MNN_FORWARD_ALL;
}

MNN::BackendConfig::PrecisionMode parseHiFTPrecision(const std::string& value) {
    if (value == "low") {
        return MNN::BackendConfig::Precision_Low;
    }
    if (value == "high") {
        return MNN::BackendConfig::Precision_High;
    }
    return MNN::BackendConfig::Precision_Normal;
}

int selectHiFTFrameBucket(int melFrames) {
    constexpr int buckets[] = {128, 256, 384, 512, 768, 1024, 1280, 1536, 2048};
    for (int bucket : buckets) {
        if (melFrames <= bucket) {
            return bucket;
        }
    }
    return ((melFrames + 255) / 256) * 256;
}

std::vector<float> padMelFrames(const std::vector<float>& input, int inputFrames,
                                int runtimeFrames) {
    if (inputFrames == runtimeFrames) {
        return input;
    }
    std::vector<float> padded(static_cast<size_t>(kChannels) * runtimeFrames);
    for (int channel = 0; channel < kChannels; ++channel) {
        const float* source = input.data() + static_cast<size_t>(channel) * inputFrames;
        float* target = padded.data() + static_cast<size_t>(channel) * runtimeFrames;
        std::copy(source, source + inputFrames, target);
        std::fill(target + inputFrames, target + runtimeFrames, source[inputFrames - 1]);
    }
    return padded;
}

struct HiFTRuntimeState {
    std::string key;
    ModelSession f0Model;
    ModelSession coreModel;
    int activeFrames = -1;
};

HiFTRuntimeState& hiftRuntimeState() {
    static HiFTRuntimeState state;
    return state;
}

void resetHiFTRuntime() {
    hiftRuntimeState() = HiFTRuntimeState{};
}

}  // namespace

int cosyVoiceHiFTMain(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " hift-f0.mnn hift-core.mnn manifest.txt threads report.jsonl"
                  << " [ready_file start_file [core_backend core_precision"
                  << " [core_cache_file [core_gpu_mode]]]]"
                  << std::endl;
        return 2;
    }
    const std::string f0ModelPath = argv[1];
    const std::string coreModelPath = argv[2];
    const std::vector<HiFTRequest> requests = readHiFTManifest(argv[3]);
    const int threads = std::max(1, std::atoi(argv[4]));
    const std::string reportPath = argv[5];
    const std::string readyPath = argc >= 7 ? argv[6] : "";
    const std::string startPath = argc >= 8 ? argv[7] : "";
    const std::string coreBackendName = argc >= 9 ? argv[8] : "cpu";
    const std::string corePrecisionName = argc >= 10 ? argv[9] : "high";
    const std::string coreCachePath = argc >= 11 ? argv[10] : "";
    const int coreGpuMode = argc >= 12
                                ? std::max(1, std::atoi(argv[11]))
                                : threads;
    const MNNForwardType coreBackend = parseHiFTBackend(coreBackendName);
    const auto corePrecision = parseHiFTPrecision(corePrecisionName);
    if (requests.empty()) {
        return 3;
    }
    if (coreBackend == MNN_FORWARD_ALL) {
        std::cerr << "Unsupported HiFT core backend: " << coreBackendName << std::endl;
        return 3;
    }

    const std::string runtimeKey = f0ModelPath + "|" + coreModelPath + "|" +
                                   std::to_string(threads) + "|" + coreBackendName + "|" +
                                   corePrecisionName + "|" + std::to_string(coreGpuMode) + "|" +
                                   coreCachePath;
    HiFTRuntimeState& runtime = hiftRuntimeState();
    double startupMs = 0.0;
    if (!runtime.f0Model.session || !runtime.coreModel.session || runtime.key != runtimeKey) {
        resetHiFTRuntime();
        const auto startupBegin = std::chrono::steady_clock::now();
        runtime.f0Model = createCpuSession(f0ModelPath, threads);
        runtime.coreModel = createModelSession(
            coreModelPath, coreBackend, corePrecision, coreGpuMode, coreCachePath);
        if (!runtime.f0Model.session || !runtime.coreModel.session) {
            resetHiFTRuntime();
            return 4;
        }
        runtime.key = runtimeKey;
        runtime.activeFrames = -1;
        startupMs = elapsedMs(startupBegin);
    }
    ModelSession& f0Model = runtime.f0Model;
    ModelSession& coreModel = runtime.coreModel;
    std::ofstream report(reportPath);
    if (!report) {
        return 5;
    }
    std::cout << std::fixed << std::setprecision(3)
              << "startup_ms=" << startupMs
              << " core_backend=" << coreBackendName
              << " core_precision=" << corePrecisionName
              << " core_gpu_mode=" << coreGpuMode
              << " vm_rss=" << currentRss() << std::endl;
    signalReadyAndWait(readyPath, startPath);

    for (size_t requestIndex = 0; requestIndex < requests.size(); ++requestIndex) {
        const HiFTRequest& request = requests[requestIndex];
        const int runtimeFrames = coreBackend == MNN_FORWARD_OPENCL
                                      ? selectHiFTFrameBucket(request.melFrames)
                                      : request.melFrames;
        const size_t sourceSamples = static_cast<size_t>(request.melFrames) * kMelHop;
        const size_t runtimeSourceSamples = static_cast<size_t>(runtimeFrames) * kMelHop;
        const size_t sourceFrames = static_cast<size_t>(runtimeFrames) * 120 + 1;
        std::vector<float> inputMel;
        std::vector<float> linearWeight;
        std::vector<float> linearBias;
        std::vector<float> noiseTable;
        if (!readFloats(joinPath(request.inputDirectory, "mel.bin"),
                        static_cast<size_t>(kChannels) * request.melFrames, inputMel) ||
            !readFloats(joinPath(request.inputDirectory, "source-linear-weight.bin"),
                        kHarmonics, linearWeight) ||
            !readFloats(joinPath(request.inputDirectory, "source-linear-bias.bin"),
                        1, linearBias)) {
            return 6;
        }
        if (!readFloats(joinPath(request.inputDirectory, "source-noise-table.bin"),
                        runtimeSourceSamples * kHarmonics, noiseTable, true)) {
            noiseTable = generateNoiseTable(runtimeSourceSamples * kHarmonics);
        }
        const std::vector<float> mel = padMelFrames(inputMel, request.melFrames, runtimeFrames);

        double resizeMs = 0.0;
        if (runtime.activeFrames != runtimeFrames) {
            const auto resizeBegin = std::chrono::steady_clock::now();
            MNN::Tensor* f0Input =
                f0Model.interpreter->getSessionInput(f0Model.session, "mel");
            f0Model.interpreter->resizeTensor(f0Input, {1, kChannels, runtimeFrames});
            f0Model.interpreter->resizeSession(f0Model.session);
            MNN::Tensor* coreMel =
                coreModel.interpreter->getSessionInput(coreModel.session, "mel");
            MNN::Tensor* coreSource =
                coreModel.interpreter->getSessionInput(coreModel.session, "source_stft");
            coreModel.interpreter->resizeTensor(coreMel, {1, kChannels, runtimeFrames});
            coreModel.interpreter->resizeTensor(
                coreSource, {1, 2 * kFftBins, static_cast<int>(sourceFrames)});
            coreModel.interpreter->resizeSession(coreModel.session);
            resizeMs = elapsedMs(resizeBegin);
            runtime.activeFrames = runtimeFrames;
        }

        const auto requestBegin = std::chrono::steady_clock::now();
        const auto f0Begin = std::chrono::steady_clock::now();
        MNN::Tensor* f0Input =
            f0Model.interpreter->getSessionInput(f0Model.session, "mel");
        if (!copyToTensor(f0Input, mel) ||
            f0Model.interpreter->runSession(f0Model.session) != MNN::NO_ERROR) {
            return 7;
        }
        std::vector<float> f0 = copyFromTensor(
            f0Model.interpreter->getSessionOutput(f0Model.session, "f0"));
        const double f0Ms = elapsedMs(f0Begin);

        const auto sourceBegin = std::chrono::steady_clock::now();
        std::vector<float> source =
            generateSource(f0, noiseTable, linearWeight, linearBias[0]);
        std::vector<float> sourceStft = stft(source);
        const double sourceMs = elapsedMs(sourceBegin);

        const auto coreBegin = std::chrono::steady_clock::now();
        MNN::Tensor* coreMel =
            coreModel.interpreter->getSessionInput(coreModel.session, "mel");
        MNN::Tensor* coreSource =
            coreModel.interpreter->getSessionInput(coreModel.session, "source_stft");
        if (!copyToTensor(coreMel, mel) || !copyToTensor(coreSource, sourceStft) ||
            coreModel.interpreter->runSession(coreModel.session) != MNN::NO_ERROR) {
            return 8;
        }
        std::vector<float> spectral = copyFromTensor(
            coreModel.interpreter->getSessionOutput(coreModel.session, "spectral"));
        const double coreMs = elapsedMs(coreBegin);

        const auto istftBegin = std::chrono::steady_clock::now();
        std::vector<float> pcm = istft(spectral, sourceFrames);
        pcm.resize(sourceSamples);
        const double istftMs = elapsedMs(istftBegin);
        const double requestMs = elapsedMs(requestBegin);
        Stats stats = calculateStats(pcm);
        double peak = std::max(std::abs(stats.min), std::abs(stats.max));
        const bool healthyBeforeWrite = stats.finite && stats.rms > 0.001 && peak <= 4.0;
        bool normalized = false;
        if (healthyBeforeWrite && peak > 0.98) {
            const float scale = static_cast<float>(0.98 / peak);
            for (float& sample : pcm) sample *= scale;
            stats = calculateStats(pcm);
            peak = std::max(std::abs(stats.min), std::abs(stats.max));
            normalized = true;
        }
        const bool wroteWave = healthyBeforeWrite &&
            writePcm16Wave(joinPath(request.outputDirectory, "hift-android.wav"), pcm);

        report << std::fixed << std::setprecision(6)
               << "{\"request\":" << requestIndex
               << ",\"melFrames\":" << request.melFrames
               << ",\"runtimeMelFrames\":" << runtimeFrames
               << ",\"audioSeconds\":"
               << static_cast<double>(sourceSamples) / kSampleRate
               << ",\"resizeMs\":" << resizeMs
               << ",\"f0Ms\":" << f0Ms
               << ",\"sourceStftMs\":" << sourceMs
               << ",\"coreMs\":" << coreMs
               << ",\"istftMs\":" << istftMs
               << ",\"requestMs\":" << requestMs
               << ",\"pcmPeak\":" << peak
               << ",\"pcmRms\":" << stats.rms
               << ",\"pcmFinite\":" << (stats.finite ? "true" : "false")
               << ",\"normalized\":" << (normalized ? "true" : "false")
               << ",\"waveWritten\":" << (wroteWave ? "true" : "false") << "}\n";
        report.flush();
        if (!wroteWave) {
            std::cerr << "pcm_health_failed=true finite=" << stats.finite
                      << " rms=" << stats.rms << " peak=" << peak
                      << " write=" << wroteWave << std::endl;
            return 9;
        }
        std::cout << std::fixed << std::setprecision(3)
                  << "request=" << requestIndex
                  << " mel_frames=" << request.melFrames
                  << " runtime_mel_frames=" << runtimeFrames
                  << " resize_ms=" << resizeMs
                  << " request_ms=" << requestMs
                  << " pcm_peak=" << peak
                  << " pcm_rms=" << stats.rms
                  << " vm_rss=" << currentRss() << std::endl;
        if (!coreCachePath.empty()) {
            coreModel.interpreter->updateCacheFile(coreModel.session);
        }
    }
    return 0;
}

#ifndef COSYVOICE_HIFT_LIBRARY
int main(int argc, char** argv) {
    return cosyVoiceHiFTMain(argc, argv);
}
#endif
