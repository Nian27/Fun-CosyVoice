#include <MNN/MNNForwardType.h>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

std::string parentDirectory(const std::string& path) {
    const auto position = path.find_last_of("/\\");
    return position == std::string::npos ? "." : path.substr(0, position);
}

bool readFloats(const std::string& path, size_t count, std::vector<float>& values) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) != count * sizeof(float)) return false;
    input.seekg(0);
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    return static_cast<bool>(input);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " model.mnn whisper-features-f32.bin frames\n";
        return 1;
    }
    const std::string modelPath = argv[1];
    const std::string featurePath = argv[2];
    const int frames = std::stoi(argv[3]);
    if (frames <= 0) return 2;

    std::vector<float> features;
    if (!readFloats(featurePath, static_cast<size_t>(128) * frames, features)) return 3;

    MNN::BackendConfig backend;
    backend.precision = MNN::BackendConfig::Precision_High;
    backend.memory = MNN::BackendConfig::Memory_Normal;
    backend.power = MNN::BackendConfig::Power_High;
    MNN::ScheduleConfig schedule;
    schedule.type = MNN_FORWARD_CPU;
    schedule.backupType = MNN_FORWARD_CPU;
    schedule.numThread = 6;
    schedule.backendConfig = &backend;

    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backend, 6);
    MNN::Express::ExecutorScope executorScope(executor);
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime(
        MNN::Express::Executor::RuntimeManager::createRuntimeManager(schedule));
    if (!runtime) return 4;
    runtime->setExternalPath(parentDirectory(modelPath), 3);
    if (std::ifstream(modelPath + ".weight").good()) {
        runtime->setExternalFile(modelPath + ".weight");
    }

    MNN::Express::Module::Config config;
    config.shapeMutable = true;
    config.dynamic = true;
    const std::vector<std::string> inputNames = {"feats", "feats_length"};
    const std::vector<std::string> outputNames = {"indices"};
    std::shared_ptr<MNN::Express::Module> module(
        MNN::Express::Module::load(inputNames, outputNames, modelPath.c_str(), runtime, &config));
    if (!module) return 5;
    const auto info = module->getInfo();
    if (!info || info->inputs.size() != 2) return 6;

    auto featureInput = MNN::Express::_Input({1, 128, frames}, info->inputs[0].order,
                                             info->inputs[0].type);
    auto lengthInput = MNN::Express::_Input({1}, info->inputs[1].order, info->inputs[1].type);
    std::copy(features.begin(), features.end(), featureInput->writeMap<float>());
    lengthInput->writeMap<int32_t>()[0] = frames;
    auto outputs = module->onForward({featureInput, lengthInput});
    if (outputs.size() != 1 || !outputs[0]->getInfo()) return 7;
    const auto outputInfo = outputs[0]->getInfo();
    const int32_t* output = outputs[0]->readMap<int32_t>();
    if (!output || outputInfo->type != halide_type_of<int32_t>() || outputInfo->size <= 0) return 8;

    std::vector<int32_t> tokens(output, output + outputInfo->size);
    const auto minimumMaximum = std::minmax_element(tokens.begin(), tokens.end());
    std::set<int32_t> unique(tokens.begin(), tokens.end());
    std::cout << "count=" << tokens.size() << " min=" << *minimumMaximum.first
              << " max=" << *minimumMaximum.second
              << " unique=" << unique.size() << "\n";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << tokens[i];
    }
    std::cout << '\n';
    return 0;
}
