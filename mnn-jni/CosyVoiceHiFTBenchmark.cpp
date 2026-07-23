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
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct InputSpec {
    std::string name;
    std::string file;
    std::vector<int> shape;
};

struct Stats {
    bool finite = true;
    float min = std::numeric_limits<float>::infinity();
    float max = -std::numeric_limits<float>::infinity();
    double mean = 0.0;
    double rms = 0.0;
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

size_t elementCount(const std::vector<int>& shape) {
    size_t count = 1;
    for (int dimension : shape) {
        count *= static_cast<size_t>(dimension);
    }
    return count;
}

Stats calculateStats(const float* values, size_t count) {
    Stats stats;
    if (count == 0) {
        stats.finite = false;
        return stats;
    }
    long double sum = 0.0;
    long double sumSquares = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const float value = values[i];
        if (!std::isfinite(value)) {
            stats.finite = false;
            continue;
        }
        stats.min = std::min(stats.min, value);
        stats.max = std::max(stats.max, value);
        sum += value;
        sumSquares += static_cast<long double>(value) * value;
    }
    stats.mean = static_cast<double>(sum / count);
    stats.rms = std::sqrt(static_cast<double>(sumSquares / count));
    return stats;
}

std::string shapeString(const MNN::Tensor& tensor) {
    std::ostringstream output;
    output << '[';
    for (int i = 0; i < tensor.dimensions(); ++i) {
        if (i) {
            output << ',';
        }
        output << tensor.length(i);
    }
    output << ']';
    return output.str();
}

bool loadInput(MNN::Tensor* tensor, const InputSpec& spec, const std::string& inputDirectory) {
    const std::string path = joinPath(inputDirectory, spec.file);
    const size_t bytes = elementCount(spec.shape) * sizeof(float);
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || static_cast<size_t>(stream.tellg()) != bytes) {
        std::cerr << "Invalid input file: " << path << " expected_bytes=" << bytes << std::endl;
        return false;
    }
    stream.seekg(0);
    MNN::Tensor host(tensor, MNN::Tensor::CAFFE);
    stream.read(reinterpret_cast<char*>(host.host<float>()), static_cast<std::streamsize>(bytes));
    if (!stream) {
        return false;
    }
    tensor->copyFromHostTensor(&host);
    return true;
}

bool writeOutput(const std::string& path, const MNN::Tensor& tensor) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }
    stream.write(reinterpret_cast<const char*>(tensor.host<float>()), tensor.size());
    return static_cast<bool>(stream);
}

double elapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
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
    if (value == "low") {
        return MNN::BackendConfig::Precision_Low;
    }
    if (value == "high") {
        return MNN::BackendConfig::Precision_High;
    }
    return MNN::BackendConfig::Precision_Normal;
}

void printMemory(const char* stage) {
    std::ifstream status("/proc/self/status");
    std::string line;
    std::cout << "stage=" << stage;
    while (std::getline(status, line)) {
        if (line.rfind("VmSize:", 0) == 0 || line.rfind("VmRSS:", 0) == 0) {
            std::cout << ' ' << line;
        }
    }
    std::cout << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0]
                  << " model.mnn f0|core input_dir output.bin mel_frames cpu|opencl"
                     " low|normal|high loops [threads_or_gpu_mode]"
                  << std::endl;
        return 2;
    }

    const std::string modelPath = argv[1];
    const std::string mode = argv[2];
    const std::string inputDirectory = argv[3];
    const std::string outputPath = argv[4];
    const int melFrames = std::max(1, std::atoi(argv[5]));
    const std::string backendName = argv[6];
    const std::string precisionName = argv[7];
    const int loops = std::max(0, std::atoi(argv[8]));
    const int threadsOrGpuMode = argc > 9 ? std::max(1, std::atoi(argv[9])) : 4;
    const int sourceFrames = melFrames * 120 + 1;
    const MNNForwardType backend = parseBackend(backendName);
    if ((mode != "f0" && mode != "core") || backend == MNN_FORWARD_ALL) {
        std::cerr << "Unsupported mode or backend" << std::endl;
        return 2;
    }

    std::vector<InputSpec> inputs = {{"mel", "mel.bin", {1, 80, melFrames}}};
    std::string outputName = "f0";
    if (mode == "core") {
        inputs.push_back({"source_stft", "source-stft.bin", {1, 18, sourceFrames}});
        outputName = "spectral";
    }

    const auto loadStart = std::chrono::steady_clock::now();
    printMemory("load_begin");
    std::shared_ptr<MNN::Interpreter> interpreter(
        MNN::Interpreter::createFromFile(modelPath.c_str()), MNN::Interpreter::destroy);
    if (!interpreter) {
        return 3;
    }

    MNN::BackendConfig backendConfig;
    backendConfig.precision = parsePrecision(precisionName);
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;
    backendConfig.power = MNN::BackendConfig::Power_High;
    MNN::ScheduleConfig schedule;
    schedule.type = backend;
    schedule.backupType = backend;
    schedule.numThread = threadsOrGpuMode;
    schedule.backendConfig = &backendConfig;
    MNN::Session* session = interpreter->createSession(schedule);
    if (!session) {
        return 4;
    }

    const auto allInputs = interpreter->getSessionInputAll(session);
    for (const InputSpec& spec : inputs) {
        const auto found = allInputs.find(spec.name);
        if (found == allInputs.end()) {
            std::cerr << "Missing input: " << spec.name << std::endl;
            return 5;
        }
        interpreter->resizeTensor(found->second, spec.shape);
    }
    interpreter->resizeSession(session);
    int resizeStatus = -1;
    interpreter->getSessionInfo(session, MNN::Interpreter::RESIZE_STATUS, &resizeStatus);
    if (resizeStatus != 0) {
        std::cerr << "Resize failed: " << resizeStatus << std::endl;
        return 6;
    }
    for (const InputSpec& spec : inputs) {
        if (!loadInput(allInputs.at(spec.name), spec, inputDirectory)) {
            return 7;
        }
    }
    printMemory("inputs_ready");

    const auto firstStart = std::chrono::steady_clock::now();
    const MNN::ErrorCode firstError = interpreter->runSession(session);
    const double firstMs = elapsedMs(firstStart);
    if (firstError != MNN::NO_ERROR) {
        std::cerr << "First inference failed: " << static_cast<int>(firstError) << std::endl;
        return 8;
    }
    MNN::Tensor* output = interpreter->getSessionOutput(session, outputName.c_str());
    if (!output) {
        return 9;
    }
    MNN::Tensor host(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(&host);

    std::vector<double> steadyTimes;
    for (int i = 0; i < loops; ++i) {
        for (const InputSpec& spec : inputs) {
            if (!loadInput(allInputs.at(spec.name), spec, inputDirectory)) {
                return 10;
            }
        }
        const auto start = std::chrono::steady_clock::now();
        const MNN::ErrorCode error = interpreter->runSession(session);
        if (error != MNN::NO_ERROR) {
            return 11;
        }
        output->copyToHostTensor(&host);
        steadyTimes.push_back(elapsedMs(start));
    }

    const size_t outputCount = static_cast<size_t>(host.elementSize());
    const Stats stats = calculateStats(host.host<float>(), outputCount);
    if (!writeOutput(outputPath, host)) {
        return 12;
    }
    std::sort(steadyTimes.begin(), steadyTimes.end());
    const double medianMs = steadyTimes.empty() ? 0.0 : steadyTimes[steadyTimes.size() / 2];
    const double averageMs = steadyTimes.empty()
                                 ? 0.0
                                 : std::accumulate(steadyTimes.begin(), steadyTimes.end(), 0.0) /
                                       steadyTimes.size();
    float memoryMb = 0.0f;
    float flopsM = 0.0f;
    int backendTypes[2] = {-1, -1};
    interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryMb);
    interpreter->getSessionInfo(session, MNN::Interpreter::FLOPS, &flopsM);
    interpreter->getSessionInfo(session, MNN::Interpreter::BACKENDS, backendTypes);

    std::ofstream metrics(outputPath + ".json");
    metrics << std::fixed << std::setprecision(6)
            << "{\n"
            << "  \"mode\": \"" << mode << "\",\n"
            << "  \"backend\": \"" << backendName << "\",\n"
            << "  \"precision\": \"" << precisionName << "\",\n"
            << "  \"melFrames\": " << melFrames << ",\n"
            << "  \"audioSeconds\": " << melFrames / 50.0 << ",\n"
            << "  \"loadMs\": " << elapsedMs(loadStart) << ",\n"
            << "  \"firstRunMs\": " << firstMs << ",\n"
            << "  \"steadyAverageMs\": " << averageMs << ",\n"
            << "  \"steadyMedianMs\": " << medianMs << ",\n"
            << "  \"memoryMb\": " << memoryMb << ",\n"
            << "  \"flopsM\": " << flopsM << ",\n"
            << "  \"primaryBackendType\": " << backendTypes[0] << ",\n"
            << "  \"backupBackendType\": " << backendTypes[1] << ",\n"
            << "  \"outputShape\": \"" << shapeString(host) << "\",\n"
            << "  \"outputElements\": " << outputCount << ",\n"
            << "  \"finite\": " << (stats.finite ? "true" : "false") << ",\n"
            << "  \"min\": " << stats.min << ",\n"
            << "  \"max\": " << stats.max << ",\n"
            << "  \"mean\": " << stats.mean << ",\n"
            << "  \"rms\": " << stats.rms << "\n"
            << "}\n";

    std::cout << std::fixed << std::setprecision(3) << "mode=" << mode << " backend=" << backendName
              << " precision=" << precisionName << " frames=" << melFrames << " shape=" << shapeString(host)
              << " finite=" << stats.finite << " first_ms=" << firstMs << " median_ms=" << medianMs
              << " memory_mb=" << memoryMb << " rms=" << stats.rms << std::endl;
    return stats.finite ? 0 : 13;
}
