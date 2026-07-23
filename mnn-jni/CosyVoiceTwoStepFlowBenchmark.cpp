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
#include <numeric>
#include <string>
#include <vector>

namespace {

struct TensorSpec {
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

struct ErrorStats {
    double mae = 0.0;
    double maxAbs = 0.0;
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
                      << " bytes" << std::endl;
        }
        return false;
    }
    input.seekg(0);
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expected));
    return static_cast<bool>(input);
}

bool writeFloats(const std::string& path, const float* values, size_t count) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        std::cerr << "Cannot create " << path << std::endl;
        return false;
    }
    output.write(reinterpret_cast<const char*>(values),
                 static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(output);
}

Stats calculateStats(const float* values, size_t count) {
    Stats stats;
    long double sum = 0.0;
    long double squares = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const float value = values[i];
        if (!std::isfinite(value)) {
            stats.finite = false;
            continue;
        }
        stats.min = std::min(stats.min, value);
        stats.max = std::max(stats.max, value);
        sum += value;
        squares += static_cast<long double>(value) * value;
    }
    if (count > 0) {
        stats.mean = static_cast<double>(sum / count);
        stats.rms = std::sqrt(static_cast<double>(squares / count));
    } else {
        stats.finite = false;
    }
    return stats;
}

ErrorStats calculateError(const std::vector<float>& reference,
                          const std::vector<float>& candidate) {
    ErrorStats result;
    long double absoluteSum = 0.0;
    long double squareSum = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double error = static_cast<double>(candidate[i]) - reference[i];
        const double absolute = std::abs(error);
        absoluteSum += absolute;
        squareSum += error * error;
        result.maxAbs = std::max(result.maxAbs, absolute);
    }
    if (!reference.empty()) {
        result.mae = static_cast<double>(absoluteSum / reference.size());
        result.rms = std::sqrt(static_cast<double>(squareSum / reference.size()));
    }
    return result;
}

bool copyToTensor(MNN::Tensor* tensor, const std::vector<float>& values) {
    if (static_cast<size_t>(tensor->elementSize()) != values.size()) {
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 10) {
        std::cerr << "Usage: " << argv[0]
                  << " model.mnn input_dir output_dir sequence_length prompt_length"
                     " backend precision loops threads_or_gpu_mode"
                  << std::endl;
        return 2;
    }

    const std::string modelPath = argv[1];
    const std::string inputDirectory = argv[2];
    const std::string outputDirectory = argv[3];
    const int sequenceLength = std::max(1, std::atoi(argv[4]));
    const int promptLength = std::max(0, std::atoi(argv[5]));
    const std::string backendName = argv[6];
    const std::string precisionName = argv[7];
    const int loops = std::max(1, std::atoi(argv[8]));
    const int threadsOrGpuMode = std::max(1, std::atoi(argv[9]));
    if (promptLength >= sequenceLength) {
        std::cerr << "prompt_length must be smaller than sequence_length" << std::endl;
        return 2;
    }

    const MNNForwardType backend = parseBackend(backendName);
    if (backend == MNN_FORWARD_ALL) {
        std::cerr << "Unsupported backend " << backendName << std::endl;
        return 2;
    }

    const std::vector<TensorSpec> specs = {
        {"x", {1, 80, sequenceLength}},
        {"mask", {1, 1, sequenceLength}},
        {"mu", {1, 80, sequenceLength}},
        {"spks", {1, 80}},
        {"cond", {1, 80, sequenceLength}},
    };
    std::vector<std::vector<float>> inputValues;
    inputValues.reserve(specs.size());
    for (const TensorSpec& spec : specs) {
        std::vector<float> values;
        if (!readFloats(joinPath(inputDirectory, std::string(spec.name) + ".bin"),
                        elementCount(spec.shape), values)) {
            return 3;
        }
        inputValues.push_back(std::move(values));
    }
    std::vector<float> initialX = inputValues[0];

    printStage("load_model_begin");
    const auto loadStart = std::chrono::steady_clock::now();
    std::shared_ptr<MNN::Interpreter> interpreter(
        MNN::Interpreter::createFromFile(modelPath.c_str()), MNN::Interpreter::destroy);
    if (!interpreter) {
        std::cerr << "Failed to load model" << std::endl;
        return 4;
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
        std::cerr << "Failed to create session" << std::endl;
        return 5;
    }
    const auto allInputs = interpreter->getSessionInputAll(session);
    for (const TensorSpec& spec : specs) {
        const auto found = allInputs.find(spec.name);
        if (found == allInputs.end()) {
            std::cerr << "Missing input " << spec.name << std::endl;
            return 6;
        }
        interpreter->resizeTensor(found->second, spec.shape);
    }
    const auto timeInput = allInputs.find("t");
    if (timeInput == allInputs.end()) {
        std::cerr << "Missing input t" << std::endl;
        return 6;
    }
    interpreter->resizeTensor(timeInput->second, {1});
    interpreter->resizeSession(session);
    printStage("load_model_done");
    const double loadMs = elapsedMs(loadStart);

    MNN::Tensor* output = interpreter->getSessionOutput(session, "estimator_out");
    if (!output) {
        std::cerr << "Missing estimator_out" << std::endl;
        return 7;
    }

    const double midpoint = 1.0 - std::cos(3.14159265358979323846 / 4.0);
    const double times[2] = {0.0, midpoint};
    const double deltas[2] = {midpoint, 1.0 - midpoint};
    std::vector<double> completeTimes;
    std::vector<float> finalX;
    std::vector<float> finalVelocity0;
    std::vector<float> finalVelocity1;

    for (int loop = 0; loop < loops; ++loop) {
        std::vector<float> x = initialX;
        const auto completeStart = std::chrono::steady_clock::now();
        for (int step = 0; step < 2; ++step) {
            printStage("flow_step_" + std::to_string(step) + "_begin");
            inputValues[0] = x;
            // Optimized dynamic graphs may alias and overwrite external inputs.
            // Restore every tensor before each estimator call.
            for (size_t i = 0; i < specs.size(); ++i) {
                if (!copyToTensor(allInputs.at(specs[i].name), inputValues[i])) {
                    std::cerr << "Failed to copy input " << specs[i].name << std::endl;
                    return 8;
                }
            }
            std::vector<float> timeValue = {static_cast<float>(times[step])};
            if (!copyToTensor(timeInput->second, timeValue)) {
                return 8;
            }
            const auto stepStart = std::chrono::steady_clock::now();
            const MNN::ErrorCode error = interpreter->runSession(session);
            const double stepMs = elapsedMs(stepStart);
            if (error != MNN::NO_ERROR) {
                std::cerr << "Flow step failed with error " << static_cast<int>(error) << std::endl;
                return 9;
            }
            std::vector<float> velocity = copyFromTensor(output);
            const Stats stats = calculateStats(velocity.data(), velocity.size());
            if (!stats.finite) {
                std::cerr << "Non-finite Flow output at step " << step << std::endl;
                return 10;
            }
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] += static_cast<float>(deltas[step]) * velocity[i];
            }
            if (loop == loops - 1) {
                if (step == 0) {
                    finalVelocity0 = std::move(velocity);
                } else {
                    finalVelocity1 = std::move(velocity);
                }
            }
            std::cout << std::fixed << std::setprecision(3)
                      << "loop=" << loop << " step=" << step << " step_ms=" << stepMs
                      << " velocity_rms=" << stats.rms << std::endl;
            printStage("flow_step_" + std::to_string(step) + "_done");
        }
        completeTimes.push_back(elapsedMs(completeStart));
        if (loop == loops - 1) {
            finalX = std::move(x);
        }
    }

    std::vector<float> referenceVelocity0;
    std::vector<float> referenceVelocity1;
    std::vector<float> referenceFinal;
    const size_t fullCount = static_cast<size_t>(80) * sequenceLength;
    const bool referenceAvailable =
        readFloats(joinPath(inputDirectory, "velocity_step0.bin"), fullCount,
                   referenceVelocity0, true) &&
        readFloats(joinPath(inputDirectory, "velocity_step1.bin"), fullCount,
                   referenceVelocity1, true) &&
        readFloats(joinPath(inputDirectory, "student_final.bin"), fullCount,
                   referenceFinal, true);
    const ErrorStats velocity0Error = referenceAvailable
                                          ? calculateError(referenceVelocity0, finalVelocity0)
                                          : ErrorStats{};
    const ErrorStats velocity1Error = referenceAvailable
                                          ? calculateError(referenceVelocity1, finalVelocity1)
                                          : ErrorStats{};
    const ErrorStats finalError = referenceAvailable
                                      ? calculateError(referenceFinal, finalX)
                                      : ErrorStats{};
    const Stats finalStats = calculateStats(finalX.data(), finalX.size());

    const size_t targetFrames = static_cast<size_t>(sequenceLength - promptLength);
    std::vector<float> targetMel(static_cast<size_t>(80) * targetFrames);
    for (size_t channel = 0; channel < 80; ++channel) {
        const float* source = finalX.data() + channel * sequenceLength + promptLength;
        std::copy(source, source + targetFrames, targetMel.data() + channel * targetFrames);
    }
    if (!writeFloats(joinPath(outputDirectory, "student_final_android.bin"), finalX.data(),
                     finalX.size()) ||
        !writeFloats(joinPath(outputDirectory, "student_target_mel_android.bin"),
                     targetMel.data(), targetMel.size())) {
        return 12;
    }

    std::sort(completeTimes.begin(), completeTimes.end());
    const double medianMs = completeTimes[completeTimes.size() / 2];
    std::ofstream metrics(joinPath(outputDirectory, "flow_two_step_android.json"));
    metrics << std::fixed << std::setprecision(6)
            << "{\n"
            << "  \"backend\": \"" << backendName << "\",\n"
            << "  \"precision\": \"" << precisionName << "\",\n"
            << "  \"sequenceLength\": " << sequenceLength << ",\n"
            << "  \"promptLength\": " << promptLength << ",\n"
            << "  \"targetLength\": " << targetFrames << ",\n"
            << "  \"loadMs\": " << loadMs << ",\n"
            << "  \"twoStepMedianMs\": " << medianMs << ",\n"
            << "  \"finite\": " << (finalStats.finite ? "true" : "false") << ",\n"
            << "  \"referenceAvailable\": " << (referenceAvailable ? "true" : "false") << ",\n"
            << "  \"finalRms\": " << finalStats.rms << ",\n"
            << "  \"step0MaxAbsError\": " << velocity0Error.maxAbs << ",\n"
            << "  \"step1MaxAbsError\": " << velocity1Error.maxAbs << ",\n"
            << "  \"finalMaxAbsError\": " << finalError.maxAbs << ",\n"
            << "  \"finalMae\": " << finalError.mae << "\n"
            << "}\n";

    std::cout << std::fixed << std::setprecision(6)
              << "two_step_median_ms=" << medianMs << " final_rms=" << finalStats.rms
              << " step0_max_abs_error=" << velocity0Error.maxAbs
              << " step1_max_abs_error=" << velocity1Error.maxAbs
              << " final_max_abs_error=" << finalError.maxAbs << std::endl;
    return finalStats.finite && (!referenceAvailable || finalError.maxAbs < 0.05) ? 0 : 13;
}
