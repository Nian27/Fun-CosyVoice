#include <MNN/ErrorCode.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
    const char* name;
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

Stats calculateStats(const float* values, size_t count);
std::string shapeString(const MNN::Tensor& tensor);

bool loadInput(MNN::Tensor* tensor, const InputSpec& spec, const std::string& inputDirectory) {
    const std::string path = joinPath(inputDirectory, std::string(spec.name) + ".bin");
    const size_t count = elementCount(spec.shape);
    const size_t bytes = count * sizeof(float);
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        std::cerr << "Cannot open input: " << path << std::endl;
        return false;
    }
    if (static_cast<size_t>(stream.tellg()) != bytes) {
        std::cerr << "Unexpected input size for " << spec.name << ": expected " << bytes << " bytes" << std::endl;
        return false;
    }
    stream.seekg(0);
    // The test vectors are exported from ONNX in contiguous NCHW order. Runtime
    // tensors may use NC4HW4 internally, so always let MNN perform the layout copy.
    MNN::Tensor host(tensor, MNN::Tensor::CAFFE);
    stream.read(reinterpret_cast<char*>(host.host<float>()), static_cast<std::streamsize>(bytes));
    if (!stream) {
        std::cerr << "Failed to read input: " << path << std::endl;
        return false;
    }
    const Stats stats = calculateStats(host.host<float>(), count);
    std::cout << std::fixed << std::setprecision(6)
              << "input=" << spec.name << " shape=" << shapeString(host)
              << " runtime_dimension_type=" << static_cast<int>(tensor->getDimensionType())
              << " finite=" << stats.finite << " min=" << stats.min << " max=" << stats.max
              << " mean=" << stats.mean << " rms=" << stats.rms << std::endl;
    tensor->copyFromHostTensor(&host);
    return true;
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

bool writeOutput(const std::string& path, const MNN::Tensor& tensor) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        std::cerr << "Cannot create output: " << path << std::endl;
        return false;
    }
    output.write(reinterpret_cast<const char*>(tensor.host<float>()), tensor.size());
    return static_cast<bool>(output);
}

MNN::ErrorCode runFirstInference(MNN::Interpreter* interpreter, MNN::Session* session,
                                 const std::string& tracePath, int traceLimit) {
    if (tracePath.empty()) {
        return interpreter->runSession(session);
    }

    std::ofstream trace(tracePath);
    if (!trace) {
        std::cerr << "Cannot create trace: " << tracePath << std::endl;
        return MNN::INPUT_DATA_ERROR;
    }
    trace << "op_index\top_name\top_type\toutput_index\ttype_code\ttype_bits\tdimension_type"
             "\tshape\telements\tfinite\tmin\tmax\tmean\trms\n";
    int opIndex = 0;
    MNN::TensorCallBackWithInfo before = [](const std::vector<MNN::Tensor*>&,
                                             const MNN::OperatorInfo*) { return true; };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& tensors,
                                            const MNN::OperatorInfo* info) {
        const int currentIndex = opIndex++;
        if (currentIndex >= traceLimit) {
            return true;
        }
        for (size_t outputIndex = 0; outputIndex < tensors.size(); ++outputIndex) {
            MNN::Tensor* tensor = tensors[outputIndex];
            const auto type = tensor->getType();
            const size_t count = static_cast<size_t>(std::max(0, tensor->elementSize()));
            trace << currentIndex << '\t' << info->name() << '\t' << info->type() << '\t'
                  << outputIndex << '\t' << static_cast<int>(type.code) << '\t'
                  << static_cast<int>(type.bits) << '\t'
                  << static_cast<int>(tensor->getDimensionType()) << '\t'
                  << shapeString(*tensor) << '\t' << count;
            if (type.code == halide_type_float && type.bits == 32 && count > 0) {
                std::shared_ptr<MNN::Tensor> host(MNN::Tensor::createHostTensorFromDevice(tensor, true));
                if (host && host->host<float>()) {
                    const Stats stats = calculateStats(host->host<float>(), count);
                    trace << '\t' << stats.finite << '\t' << stats.min << '\t' << stats.max
                          << '\t' << stats.mean << '\t' << stats.rms;
                } else {
                    trace << "\t0\tnan\tnan\tnan\tnan";
                }
            } else {
                trace << "\tNA\tNA\tNA\tNA\tNA";
            }
            trace << '\n';
        }
        return true;
    };
    return interpreter->runSessionWithCallBackInfo(session, before, after, true);
}

double elapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void printStage(const char* stage) {
    std::ifstream status("/proc/self/status");
    std::string line;
    std::string vmSize;
    std::string vmRss;
    while (std::getline(status, line)) {
        if (line.rfind("VmSize:", 0) == 0) {
            vmSize = line.substr(7);
        } else if (line.rfind("VmRSS:", 0) == 0) {
            vmRss = line.substr(6);
        }
    }
    std::cout << "stage=" << stage;
    if (!vmSize.empty()) {
        std::cout << " vm_size=" << vmSize;
    }
    if (!vmRss.empty()) {
        std::cout << " vm_rss=" << vmRss;
    }
    std::cout << std::endl;
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

}  // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " model.mnn input_dir output.bin seq_len cpu|opencl low|normal|high loops"
                     " [threads_or_gpu_mode] [batch_size] [trace.tsv] [trace_op_limit]"
                  << std::endl;
        return 2;
    }

    const std::string modelPath = argv[1];
    const std::string inputDirectory = argv[2];
    const std::string outputPath = argv[3];
    const int sequenceLength = std::max(1, std::atoi(argv[4]));
    const std::string backendName = argv[5];
    const std::string precisionName = argv[6];
    const int loops = std::max(0, std::atoi(argv[7]));
    const int threadsOrGpuMode = argc > 8 ? std::max(1, std::atoi(argv[8])) : 4;
    const int batchSize = argc > 9 ? std::max(1, std::atoi(argv[9])) : 2;
    const std::string tracePath = argc > 10 ? argv[10] : "";
    const int traceLimit = argc > 11 ? std::max(1, std::atoi(argv[11])) : 200;
    const MNNForwardType backend = parseBackend(backendName);
    if (backend == MNN_FORWARD_ALL) {
        std::cerr << "Unsupported backend: " << backendName << std::endl;
        return 2;
    }

    const std::vector<InputSpec> inputs = {
        {"x", {batchSize, 80, sequenceLength}},
        {"mask", {batchSize, 1, sequenceLength}},
        {"mu", {batchSize, 80, sequenceLength}},
        {"t", {batchSize}},
        {"spks", {batchSize, 80}},
        {"cond", {batchSize, 80, sequenceLength}},
    };

    const auto loadStart = std::chrono::steady_clock::now();
    printStage("load_model_begin");
    std::shared_ptr<MNN::Interpreter> interpreter(
        MNN::Interpreter::createFromFile(modelPath.c_str()), MNN::Interpreter::destroy);
    if (!interpreter) {
        std::cerr << "Failed to load model" << std::endl;
        return 3;
    }
    printStage("load_model_done");
    MNN::BackendConfig backendConfig;
    backendConfig.precision = parsePrecision(precisionName);
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;
    backendConfig.power = MNN::BackendConfig::Power_High;

    MNN::ScheduleConfig schedule;
    schedule.type = backend;
    schedule.backupType = backend;
    schedule.numThread = threadsOrGpuMode;
    schedule.backendConfig = &backendConfig;

    printStage("create_session_begin");
    MNN::Session* session = interpreter->createSession(schedule);
    if (!session) {
        std::cerr << "Failed to create session for backend " << backendName << std::endl;
        return 4;
    }
    printStage("create_session_done");

    const auto allInputs = interpreter->getSessionInputAll(session);
    for (const InputSpec& spec : inputs) {
        const auto found = allInputs.find(spec.name);
        if (found == allInputs.end()) {
            std::cerr << "Missing model input: " << spec.name << std::endl;
            return 5;
        }
        interpreter->resizeTensor(found->second, spec.shape);
    }
    printStage("resize_session_begin");
    interpreter->resizeSession(session);
    printStage("resize_session_done");

    int resizeStatus = -1;
    interpreter->getSessionInfo(session, MNN::Interpreter::RESIZE_STATUS, &resizeStatus);
    if (resizeStatus != 0) {
        std::cerr << "Session resize failed with status " << resizeStatus << std::endl;
        return 6;
    }

    for (const InputSpec& spec : inputs) {
        if (!loadInput(allInputs.at(spec.name), spec, inputDirectory)) {
            return 7;
        }
    }
    printStage("inputs_loaded");

    float memoryMb = 0.0f;
    float flopsM = 0.0f;
    int backendTypes[2] = {-1, -1};
    interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryMb);
    interpreter->getSessionInfo(session, MNN::Interpreter::FLOPS, &flopsM);
    interpreter->getSessionInfo(session, MNN::Interpreter::BACKENDS, backendTypes);
    const double loadMs = elapsedMs(loadStart);

    printStage("first_run_begin");
    const auto firstStart = std::chrono::steady_clock::now();
    const MNN::ErrorCode firstError = runFirstInference(interpreter.get(), session, tracePath, traceLimit);
    const double firstMs = elapsedMs(firstStart);
    if (firstError != MNN::NO_ERROR) {
        std::cerr << "First inference failed with error " << static_cast<int>(firstError) << std::endl;
        return 8;
    }
    printStage("first_run_done");

    MNN::Tensor* outputTensor = interpreter->getSessionOutput(session, "estimator_out");
    if (!outputTensor) {
        std::cerr << "Missing estimator_out" << std::endl;
        return 9;
    }
    MNN::Tensor outputHost(outputTensor, MNN::Tensor::CAFFE);
    outputTensor->copyToHostTensor(&outputHost);

    std::vector<double> steadyTimes;
    steadyTimes.reserve(static_cast<size_t>(loops));
    for (int i = 0; i < loops; ++i) {
        // Optimized dynamic graphs may reuse storage that aliases session inputs.
        // Restore every external input before treating another run as independent.
        for (const InputSpec& spec : inputs) {
            if (!loadInput(allInputs.at(spec.name), spec, inputDirectory)) {
                return 10;
            }
        }
        const auto start = std::chrono::steady_clock::now();
        const MNN::ErrorCode error = interpreter->runSession(session);
        if (error != MNN::NO_ERROR) {
            std::cerr << "Inference loop " << i << " failed with error " << static_cast<int>(error) << std::endl;
            return 11;
        }
        outputTensor->copyToHostTensor(&outputHost);
        steadyTimes.push_back(elapsedMs(start));
    }

    const size_t outputCount = static_cast<size_t>(outputHost.elementSize());
    const Stats stats = calculateStats(outputHost.host<float>(), outputCount);
    if (!writeOutput(outputPath, outputHost)) {
        return 12;
    }
    std::sort(steadyTimes.begin(), steadyTimes.end());
    const double medianMs = steadyTimes.empty() ? 0.0 : steadyTimes[steadyTimes.size() / 2];
    const double averageMs = steadyTimes.empty()
                                 ? 0.0
                                 : std::accumulate(steadyTimes.begin(), steadyTimes.end(), 0.0) /
                                       steadyTimes.size();

    std::ofstream metrics(outputPath + ".json");
    metrics << std::fixed << std::setprecision(6)
            << "{\n"
            << "  \"backend\": \"" << backendName << "\",\n"
            << "  \"precision\": \"" << precisionName << "\",\n"
            << "  \"sequenceLength\": " << sequenceLength << ",\n"
            << "  \"batchSize\": " << batchSize << ",\n"
            << "  \"loadMs\": " << loadMs << ",\n"
            << "  \"firstRunMs\": " << firstMs << ",\n"
            << "  \"steadyAverageMs\": " << averageMs << ",\n"
            << "  \"steadyMedianMs\": " << medianMs << ",\n"
            << "  \"memoryMb\": " << memoryMb << ",\n"
            << "  \"flopsM\": " << flopsM << ",\n"
            << "  \"primaryBackendType\": " << backendTypes[0] << ",\n"
            << "  \"backupBackendType\": " << backendTypes[1] << ",\n"
            << "  \"outputShape\": \"" << shapeString(outputHost) << "\",\n"
            << "  \"outputElements\": " << outputCount << ",\n"
            << "  \"finite\": " << (stats.finite ? "true" : "false") << ",\n"
            << "  \"min\": " << stats.min << ",\n"
            << "  \"max\": " << stats.max << ",\n"
            << "  \"mean\": " << stats.mean << ",\n"
            << "  \"rms\": " << stats.rms << "\n"
            << "}\n";

    std::cout << std::fixed << std::setprecision(3)
              << "backend=" << backendName << " precision=" << precisionName << " batch=" << batchSize
              << " seq=" << sequenceLength
              << " shape=" << shapeString(outputHost) << " finite=" << stats.finite
              << " load_ms=" << loadMs << " first_ms=" << firstMs << " median_ms=" << medianMs
              << " memory_mb=" << memoryMb << " rms=" << stats.rms << std::endl;
    return stats.finite ? 0 : 13;
}
