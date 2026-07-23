#include <MNN/ErrorCode.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Request {
    std::string inputDirectory;
    std::string outputDirectory;
    int sequenceLength = 0;
    int promptLength = 0;
    int runtimeSequenceLength = 0;
};

struct TensorSpec {
    const char* name;
    std::vector<int> shape;
};

struct Stats {
    bool finite = true;
    double rms = 0.0;
};

double elapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

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

std::vector<Request> readManifest(const std::string& path) {
    std::ifstream input(path);
    std::vector<Request> requests;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::istringstream fields(line);
        Request request;
        if (!(fields >> request.inputDirectory >> request.outputDirectory >>
              request.sequenceLength >> request.promptLength)) {
            continue;
        }
        if (!(fields >> request.runtimeSequenceLength)) {
            request.runtimeSequenceLength = request.sequenceLength;
        }
        if (request.sequenceLength > request.promptLength &&
            request.runtimeSequenceLength >= request.sequenceLength &&
            request.promptLength >= 0) {
            requests.push_back(request);
        }
    }
    return requests;
}

size_t elementCount(const std::vector<int>& shape) {
    size_t count = 1;
    for (int value : shape) {
        count *= static_cast<size_t>(value);
    }
    return count;
}

bool readFloats(const std::string& path, size_t count, std::vector<float>& values) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) != count * sizeof(float)) {
        std::cerr << "invalid_input=" << path << std::endl;
        return false;
    }
    input.seekg(0);
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    return static_cast<bool>(input);
}

bool writeFloats(const std::string& path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    return static_cast<bool>(output);
}

bool copyToTensor(MNN::Tensor* tensor, const std::vector<float>& values) {
    if (!tensor || static_cast<size_t>(tensor->elementSize()) != values.size()) {
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

Stats calculateStats(const std::vector<float>& values) {
    Stats stats;
    long double squares = 0.0;
    for (float value : values) {
        if (!std::isfinite(value)) {
            stats.finite = false;
            continue;
        }
        squares += static_cast<long double>(value) * value;
    }
    if (values.empty()) {
        stats.finite = false;
    } else {
        stats.rms = std::sqrt(static_cast<double>(squares / values.size()));
    }
    return stats;
}

MNNForwardType parseBackend(const std::string& value) {
    if (value == "cpu") {
        return MNN_FORWARD_CPU;
    }
    if (value == "opencl") {
        return MNN_FORWARD_OPENCL;
    }
    return MNN_FORWARD_ALL;
}

MNN::BackendConfig::PrecisionMode parsePrecision(const std::string& value) {
    if (value == "high") {
        return MNN::BackendConfig::Precision_High;
    }
    if (value == "low") {
        return MNN::BackendConfig::Precision_Low;
    }
    return MNN::BackendConfig::Precision_Normal;
}

struct FlowRuntimeState {
    std::string key;
    std::shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session* session = nullptr;
    int activeSequenceLength = -1;
};

FlowRuntimeState& flowRuntimeState() {
    static FlowRuntimeState state;
    return state;
}

void resetFlowRuntime() {
    flowRuntimeState() = FlowRuntimeState{};
}

}  // namespace

int cosyVoiceFlowMain(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " model.mnn manifest.txt backend precision threads report.jsonl"
                  << " [ready_file start_file [warmup_count [cache_file]]]"
                  << std::endl;
        return 2;
    }
    const std::string modelPath = argv[1];
    const std::vector<Request> requests = readManifest(argv[2]);
    const std::string backendName = argv[3];
    const std::string precisionName = argv[4];
    const int threads = std::max(1, std::atoi(argv[5]));
    const std::string reportPath = argv[6];
    const std::string readyPath = argc >= 8 ? argv[7] : "";
    const std::string startPath = argc >= 9 ? argv[8] : "";
    const size_t warmupCount = argc >= 10
                                   ? static_cast<size_t>(std::max(0, std::atoi(argv[9])))
                                   : 0;
    const std::string cachePath = argc >= 11 ? argv[10] : "";
    if (requests.empty()) {
        std::cerr << "manifest_empty=true" << std::endl;
        return 3;
    }
    const MNNForwardType backend = parseBackend(backendName);
    if (backend == MNN_FORWARD_ALL) {
        return 3;
    }

    const std::string runtimeKey = modelPath + "|" + backendName + "|" +
                                   precisionName + "|" + std::to_string(threads) +
                                   "|" + cachePath;
    FlowRuntimeState& runtime = flowRuntimeState();
    double startupMs = 0.0;
    if (!runtime.interpreter || !runtime.session || runtime.key != runtimeKey) {
        resetFlowRuntime();
        const auto startupBegin = std::chrono::steady_clock::now();
        runtime.interpreter.reset(
            MNN::Interpreter::createFromFile(modelPath.c_str()), MNN::Interpreter::destroy);
        if (!runtime.interpreter) {
            return 4;
        }
        if (!cachePath.empty()) {
            runtime.interpreter->setCacheFile(cachePath.c_str());
        }
        MNN::BackendConfig backendConfig;
        backendConfig.precision = parsePrecision(precisionName);
        backendConfig.memory = MNN::BackendConfig::Memory_Normal;
        backendConfig.power = MNN::BackendConfig::Power_High;
        MNN::ScheduleConfig schedule;
        schedule.type = backend;
        schedule.backupType = backend;
        schedule.numThread = threads;
        schedule.backendConfig = &backendConfig;
        runtime.session = runtime.interpreter->createSession(schedule);
        if (!runtime.session) {
            resetFlowRuntime();
            return 5;
        }
        runtime.key = runtimeKey;
        runtime.activeSequenceLength = -1;
        startupMs = elapsedMs(startupBegin);
    }
    const std::shared_ptr<MNN::Interpreter>& interpreter = runtime.interpreter;
    MNN::Session* session = runtime.session;
    std::ofstream report(reportPath);
    if (!report) {
        return 6;
    }
    std::cout << std::fixed << std::setprecision(3)
              << "startup_ms=" << startupMs << " vm_rss=" << currentRss() << std::endl;

    int& activeSequenceLength = runtime.activeSequenceLength;
    for (size_t requestIndex = 0; requestIndex < requests.size(); ++requestIndex) {
        if (requestIndex == warmupCount) {
            signalReadyAndWait(readyPath, startPath);
        }
        const Request& request = requests[requestIndex];
        const std::vector<TensorSpec> specs = {
            {"x", {1, 80, request.runtimeSequenceLength}},
            {"mask", {1, 1, request.runtimeSequenceLength}},
            {"mu", {1, 80, request.runtimeSequenceLength}},
            {"spks", {1, 80}},
            {"cond", {1, 80, request.runtimeSequenceLength}},
        };
        std::vector<std::vector<float>> inputs;
        for (size_t specIndex = 0; specIndex < specs.size(); ++specIndex) {
            const TensorSpec& spec = specs[specIndex];
            std::vector<int> sourceShape = spec.shape;
            if (std::string(spec.name) != "spks") {
                sourceShape.back() = request.sequenceLength;
            }
            std::vector<float> sourceValues;
            if (!readFloats(joinPath(request.inputDirectory,
                                     std::string(spec.name) + ".bin"),
                            elementCount(sourceShape), sourceValues)) {
                return 7;
            }
            if (request.runtimeSequenceLength == request.sequenceLength ||
                std::string(spec.name) == "spks") {
                inputs.push_back(std::move(sourceValues));
                continue;
            }
            std::vector<float> padded(elementCount(spec.shape), 0.0f);
            if (std::string(spec.name) == "mask") {
                std::copy(sourceValues.begin(), sourceValues.end(), padded.begin());
            } else {
                for (int channel = 0; channel < 80; ++channel) {
                    std::copy(sourceValues.begin() +
                                  static_cast<size_t>(channel) * request.sequenceLength,
                              sourceValues.begin() +
                                  static_cast<size_t>(channel + 1) * request.sequenceLength,
                              padded.begin() +
                                  static_cast<size_t>(channel) * request.runtimeSequenceLength);
                }
            }
            inputs.push_back(std::move(padded));
        }
        const std::vector<float> initialX = inputs[0];

        double resizeMs = 0.0;
        if (activeSequenceLength != request.runtimeSequenceLength) {
            const auto resizeBegin = std::chrono::steady_clock::now();
            const auto allInputs = interpreter->getSessionInputAll(session);
            for (const TensorSpec& spec : specs) {
                interpreter->resizeTensor(allInputs.at(spec.name), spec.shape);
            }
            interpreter->resizeTensor(allInputs.at("t"), {1});
            interpreter->resizeSession(session);
            resizeMs = elapsedMs(resizeBegin);
            activeSequenceLength = request.runtimeSequenceLength;
        }

        const auto allInputs = interpreter->getSessionInputAll(session);
        MNN::Tensor* output = interpreter->getSessionOutput(session, "estimator_out");
        const double midpoint = 1.0 - std::cos(3.14159265358979323846 / 4.0);
        const double times[2] = {0.0, midpoint};
        const double deltas[2] = {midpoint, 1.0 - midpoint};
        std::vector<float> x = initialX;
        const auto inferenceBegin = std::chrono::steady_clock::now();
        for (int step = 0; step < 2; ++step) {
            inputs[0] = x;
            for (size_t i = 0; i < specs.size(); ++i) {
                if (!copyToTensor(allInputs.at(specs[i].name), inputs[i])) {
                    return 8;
                }
            }
            if (!copyToTensor(allInputs.at("t"),
                              {static_cast<float>(times[step])}) ||
                interpreter->runSession(session) != MNN::NO_ERROR) {
                return 9;
            }
            const std::vector<float> velocity = copyFromTensor(output);
            const Stats velocityStats = calculateStats(velocity);
            if (!velocityStats.finite) {
                return 10;
            }
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] += static_cast<float>(deltas[step]) * velocity[i];
            }
        }
        const double inferenceMs = elapsedMs(inferenceBegin);
        const Stats finalStats = calculateStats(x);
        if (!finalStats.finite) {
            return 10;
        }

        const int targetFrames = request.sequenceLength - request.promptLength;
        std::vector<float> targetMel(static_cast<size_t>(80) * targetFrames);
        for (int channel = 0; channel < 80; ++channel) {
            const float* source = x.data() +
                                  static_cast<size_t>(channel) * request.runtimeSequenceLength +
                                  request.promptLength;
            std::copy(source, source + targetFrames,
                      targetMel.data() + static_cast<size_t>(channel) * targetFrames);
        }
        if (!writeFloats(joinPath(request.outputDirectory, "student_target_mel_android.bin"),
                         targetMel)) {
            return 11;
        }

        report << std::fixed << std::setprecision(6)
               << "{\"request\":" << requestIndex
               << ",\"sequenceLength\":" << request.sequenceLength
               << ",\"runtimeSequenceLength\":" << request.runtimeSequenceLength
               << ",\"targetFrames\":" << targetFrames
               << ",\"resizeMs\":" << resizeMs
               << ",\"inferenceMs\":" << inferenceMs
               << ",\"finalRms\":" << finalStats.rms << "}\n";
        report.flush();
        std::cout << std::fixed << std::setprecision(3)
                  << "request=" << requestIndex
                  << " sequence_length=" << request.sequenceLength
                  << " runtime_sequence_length=" << request.runtimeSequenceLength
                  << " resize_ms=" << resizeMs
                  << " inference_ms=" << inferenceMs
                  << " final_rms=" << finalStats.rms
                  << " vm_rss=" << currentRss() << std::endl;
        if (!cachePath.empty()) {
            interpreter->updateCacheFile(session);
        }
    }
    if (warmupCount >= requests.size()) {
        signalReadyAndWait(readyPath, startPath);
    }
    return 0;
}

#ifndef COSYVOICE_FLOW_LIBRARY
int main(int argc, char** argv) {
    return cosyVoiceFlowMain(argc, argv);
}
#endif
