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
#include <string>
#include <vector>

namespace {

std::string joinPath(const std::string& directory, const std::string& file) {
    if (!directory.empty() && directory.back() != '/' && directory.back() != '\\') {
#ifdef _WIN32
        return directory + "\\" + file;
#else
        return directory + "/" + file;
#endif
    }
    return directory + file;
}

std::vector<int> readCsv(const std::string& path) {
    std::ifstream input(path);
    std::vector<int> values;
    int value;
    while (input >> value) {
        values.push_back(value);
        if (input.peek() == ',') {
            input.ignore();
        }
    }
    return values;
}

bool readFloats(const std::string& path, size_t count, std::vector<float>& values) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) != count * sizeof(float)) {
        std::cerr << "Invalid float input " << path << std::endl;
        return false;
    }
    input.seekg(0);
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
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

double elapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " conditioner.mnn voice_dir target_tokens.csv output_dir prompt_tokens prompt_frames"
                  << std::endl;
        return 2;
    }
    const std::string modelPath = argv[1];
    const std::string voiceDirectory = argv[2];
    const std::string targetTokenPath = argv[3];
    const std::string outputDirectory = argv[4];
    const int expectedPromptTokens = std::max(1, std::atoi(argv[5]));
    const int promptFrames = std::max(1, std::atoi(argv[6]));

    std::vector<int> promptTokens = readCsv(joinPath(voiceDirectory, "prompt-speech-tokens.csv"));
    std::vector<int> targetTokens = readCsv(targetTokenPath);
    if (static_cast<int>(promptTokens.size()) != expectedPromptTokens || targetTokens.empty()) {
        std::cerr << "Invalid prompt or target token count" << std::endl;
        return 3;
    }
    std::vector<int64_t> allTokens;
    allTokens.reserve(promptTokens.size() + targetTokens.size());
    for (int value : promptTokens) {
        allTokens.push_back(value);
    }
    for (int value : targetTokens) {
        allTokens.push_back(value);
    }
    const int sequenceLength = static_cast<int>(allTokens.size()) * 2;

    const auto loadStart = std::chrono::steady_clock::now();
    std::shared_ptr<MNN::Interpreter> interpreter(
        MNN::Interpreter::createFromFile(modelPath.c_str()), MNN::Interpreter::destroy);
    if (!interpreter) {
        return 4;
    }
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;
    backendConfig.power = MNN::BackendConfig::Power_High;
    MNN::ScheduleConfig schedule;
    schedule.type = MNN_FORWARD_CPU;
    schedule.backupType = MNN_FORWARD_CPU;
    schedule.numThread = 4;
    schedule.backendConfig = &backendConfig;
    MNN::Session* session = interpreter->createSession(schedule);
    if (!session) {
        return 5;
    }
    MNN::Tensor* tokenInput = interpreter->getSessionInput(session, "tokens");
    interpreter->resizeTensor(tokenInput, {1, static_cast<int>(allTokens.size())});
    interpreter->resizeSession(session);
    MNN::Tensor tokenHost(tokenInput, MNN::Tensor::CAFFE);
    const halide_type_t tokenType = tokenInput->getType();
    if (tokenType.code != halide_type_int) {
        std::cerr << "Unexpected token tensor type code=" << static_cast<int>(tokenType.code)
                  << " bits=" << static_cast<int>(tokenType.bits) << std::endl;
        return 6;
    }
    if (tokenType.bits == 32) {
        for (size_t i = 0; i < allTokens.size(); ++i) {
            tokenHost.host<int32_t>()[i] = static_cast<int32_t>(allTokens[i]);
        }
    } else if (tokenType.bits == 64) {
        std::copy(allTokens.begin(), allTokens.end(), tokenHost.host<int64_t>());
    } else {
        std::cerr << "Unsupported token tensor bits=" << static_cast<int>(tokenType.bits)
                  << std::endl;
        return 6;
    }
    tokenInput->copyFromHostTensor(&tokenHost);
    const auto inferenceStart = std::chrono::steady_clock::now();
    if (interpreter->runSession(session) != MNN::NO_ERROR) {
        return 6;
    }
    const double inferenceMs = elapsedMs(inferenceStart);
    MNN::Tensor* muOutput = interpreter->getSessionOutput(session, "mu");
    MNN::Tensor muHost(muOutput, MNN::Tensor::CAFFE);
    muOutput->copyToHostTensor(&muHost);
    std::vector<float> mu(muHost.host<float>(), muHost.host<float>() + muHost.elementSize());
    if (mu.size() != static_cast<size_t>(80 * sequenceLength)) {
        std::cerr << "Unexpected mu size " << mu.size() << std::endl;
        return 7;
    }

    std::vector<float> promptCond;
    std::vector<float> spks;
    std::vector<float> randNoise;
    if (!readFloats(joinPath(voiceDirectory, "prompt-cond.bin"), 80 * promptFrames,
                    promptCond) ||
        !readFloats(joinPath(voiceDirectory, "spks.bin"), 80, spks) ||
        !readFloats(joinPath(voiceDirectory, "rand-noise.bin"), 80 * 15000,
                    randNoise)) {
        return 8;
    }
    std::vector<float> x(static_cast<size_t>(80) * sequenceLength);
    std::vector<float> cond(static_cast<size_t>(80) * sequenceLength, 0.0f);
    std::vector<float> mask(sequenceLength, 1.0f);
    for (int channel = 0; channel < 80; ++channel) {
        std::copy(randNoise.begin() + static_cast<size_t>(channel) * 15000,
                  randNoise.begin() + static_cast<size_t>(channel) * 15000 + sequenceLength,
                  x.begin() + static_cast<size_t>(channel) * sequenceLength);
        std::copy(promptCond.begin() + static_cast<size_t>(channel) * promptFrames,
                  promptCond.begin() + static_cast<size_t>(channel + 1) * promptFrames,
                  cond.begin() + static_cast<size_t>(channel) * sequenceLength);
    }
    if (!writeFloats(joinPath(outputDirectory, "x.bin"), x) ||
        !writeFloats(joinPath(outputDirectory, "mask.bin"), mask) ||
        !writeFloats(joinPath(outputDirectory, "mu.bin"), mu) ||
        !writeFloats(joinPath(outputDirectory, "spks.bin"), spks) ||
        !writeFloats(joinPath(outputDirectory, "cond.bin"), cond)) {
        return 9;
    }

    double maxAbsError = 0.0;
    std::vector<float> referenceMu;
    const bool hasReference = readFloats(joinPath(voiceDirectory, "mu-pytorch.bin"),
                                         mu.size(), referenceMu);
    if (hasReference) {
        for (size_t i = 0; i < mu.size(); ++i) {
            maxAbsError = std::max(maxAbsError,
                                   std::abs(static_cast<double>(mu[i]) - referenceMu[i]));
        }
    }
    const double loadMs = elapsedMs(loadStart);
    std::ofstream report(joinPath(outputDirectory, "conditioner-android.json"));
    report << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"promptTokens\": " << promptTokens.size() << ",\n"
           << "  \"targetTokens\": " << targetTokens.size() << ",\n"
           << "  \"sequenceLength\": " << sequenceLength << ",\n"
           << "  \"targetFrames\": " << sequenceLength - promptFrames << ",\n"
           << "  \"loadMs\": " << loadMs << ",\n"
           << "  \"inferenceMs\": " << inferenceMs << ",\n"
           << "  \"referenceAvailable\": " << (hasReference ? "true" : "false") << ",\n"
           << "  \"maxAbsError\": " << maxAbsError << "\n"
           << "}\n";
    std::cout << std::fixed << std::setprecision(6)
              << "prompt_tokens=" << promptTokens.size()
              << " target_tokens=" << targetTokens.size()
              << " sequence_length=" << sequenceLength
              << " token_bits=" << static_cast<int>(tokenType.bits)
              << " inference_ms=" << inferenceMs
              << " reference_available=" << hasReference
              << " max_abs_error=" << maxAbsError << std::endl;
    return !hasReference || maxAbsError < 0.01 ? 0 : 10;
}
