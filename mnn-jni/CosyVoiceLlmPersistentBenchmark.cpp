#include <llm/llm.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using MNN::Transformer::Llm;

namespace {

constexpr int kSpeechOffset = 151924;
constexpr int kSpeechTokenMax = kSpeechOffset + 6560;
constexpr int kSpeechEos = kSpeechOffset + 6562;

double elapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

std::vector<std::string> readLines(const std::string& path) {
    std::ifstream input(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

std::vector<int> readPromptSpeechTokens(const std::string& path) {
    std::ifstream input(path);
    std::vector<int> tokens;
    int token = 0;
    while (input >> token) {
        tokens.push_back(token);
        if (input.peek() == ',') {
            input.ignore();
        }
    }
    return tokens;
}

bool isSpeechOutput(int token) {
    return (token >= kSpeechOffset && token <= kSpeechTokenMax) ||
           token == kSpeechEos;
}

bool writeSpeechTokens(const std::string& path, const std::vector<int>& outputIds) {
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    bool first = true;
    for (int token : outputIds) {
        if (token < kSpeechOffset || token > kSpeechTokenMax) {
            continue;
        }
        if (!first) {
            output << ',';
        }
        output << token - kSpeechOffset;
        first = false;
    }
    output << '\n';
    return static_cast<bool>(output);
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

struct LlmRuntimeState {
    std::string configPath;
    std::unique_ptr<Llm> llm;
};

LlmRuntimeState& llmRuntimeState() {
    static LlmRuntimeState state;
    return state;
}

void resetLlmRuntime() {
    llmRuntimeState() = LlmRuntimeState{};
}

}  // namespace

int cosyVoiceLlmMain(int argc, const char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " config.json prompts.txt max_tokens prompt_speech_tokens.csv output_dir"
                  << std::endl;
        return 2;
    }

    const std::string configPath = argv[1];
    const std::string promptsPath = argv[2];
    const int maxTokens = std::max(1, std::atoi(argv[3]));
    const std::string promptTokenPath = argv[4];
    const std::string outputDirectory = argv[5];
    const std::string readyPath = argc >= 7 ? argv[6] : "";
    const std::string startPath = argc >= 8 ? argv[7] : "";
    const bool appendPromptSpeechTokens = argc < 9 || std::string(argv[8]) != "0";
    const std::vector<std::string> prompts = readLines(promptsPath);
    const std::vector<int> promptSpeechTokens = readPromptSpeechTokens(promptTokenPath);
    if (prompts.empty() || (appendPromptSpeechTokens && promptSpeechTokens.empty())) {
        std::cerr << "prompts_or_prompt_speech_tokens_empty=true" << std::endl;
        return 3;
    }

    LlmRuntimeState& runtime = llmRuntimeState();
    double loadMs = 0.0;
    if (!runtime.llm || runtime.configPath != configPath) {
        resetLlmRuntime();
        const auto loadStart = std::chrono::steady_clock::now();
        runtime.llm.reset(Llm::createLLM(configPath));
        if (!runtime.llm || !runtime.llm->load()) {
            resetLlmRuntime();
            std::cerr << "load=false" << std::endl;
            return 4;
        }
        runtime.configPath = configPath;
        loadMs = elapsedMs(loadStart);
    }
    Llm* llm = runtime.llm.get();
    std::cout << std::fixed << std::setprecision(3)
              << "load_ms=" << loadMs << " vm_rss=" << currentRss() << std::endl;
    signalReadyAndWait(readyPath, startPath);

    std::ofstream metrics(joinPath(outputDirectory, "llm-persistent.jsonl"));
    if (!metrics) {
        std::cerr << "cannot_create_metrics=true" << std::endl;
        return 5;
    }

    for (size_t request = 0; request < prompts.size(); ++request) {
        llm->reset();
        std::string prompt = llm->apply_chat_template(prompts[request]);
        if (prompt.empty()) {
            prompt = prompts[request];
        }
        std::vector<int> inputIds = llm->tokenizer_encode(prompt);
        if (appendPromptSpeechTokens) {
            for (int token : promptSpeechTokens) {
                inputIds.push_back(token + kSpeechOffset);
            }
        }

        const auto* beforeContext = llm->getContext();
        const int64_t prefillBefore = beforeContext->prefill_us;
        const int64_t decodeBefore = beforeContext->decode_us;
        const auto requestStart = std::chrono::steady_clock::now();
        const std::vector<int> outputIds = llm->generate(inputIds, maxTokens);
        const double wallMs = elapsedMs(requestStart);
        const auto* context = llm->getContext();

        int validSpeech = 0;
        int invalid = 0;
        int speechTokens = 0;
        for (int token : outputIds) {
            if (isSpeechOutput(token)) {
                ++validSpeech;
                if (token != kSpeechEos) {
                    ++speechTokens;
                }
            } else {
                ++invalid;
            }
        }
        const std::string tokenFile =
            joinPath(outputDirectory, "speech-tokens-" + std::to_string(request) + ".csv");
        if (!writeSpeechTokens(tokenFile, outputIds)) {
            std::cerr << "cannot_write_tokens=" << tokenFile << std::endl;
            return 6;
        }

        const int64_t prefillDelta = context->prefill_us >= prefillBefore
                                         ? context->prefill_us - prefillBefore
                                         : context->prefill_us;
        const int64_t decodeDelta = context->decode_us >= decodeBefore
                                        ? context->decode_us - decodeBefore
                                        : context->decode_us;
        const double prefillMs = prefillDelta / 1000.0;
        const double decodeMs = decodeDelta / 1000.0;
        const double tokensPerSecond = decodeMs > 0.0
                                           ? context->gen_seq_len * 1000.0 / decodeMs
                                           : 0.0;
        metrics << std::fixed << std::setprecision(6)
                << "{\"request\":" << request
                << ",\"inputTokens\":" << context->prompt_len
                << ",\"generatedTokens\":" << context->gen_seq_len
                << ",\"speechTokens\":" << speechTokens
                << ",\"validSpeechOutputs\":" << validSpeech
                << ",\"invalidOutputs\":" << invalid
                << ",\"promptSpeechTokensAppended\":"
                << (appendPromptSpeechTokens ? "true" : "false")
                << ",\"loadMs\":" << loadMs
                << ",\"prefillMs\":" << prefillMs
                << ",\"decodeMs\":" << decodeMs
                << ",\"wallMs\":" << wallMs
                << ",\"tokensPerSecond\":" << tokensPerSecond << "}\n";
        metrics.flush();

        std::cout << std::fixed << std::setprecision(3)
                  << "request=" << request
                  << " speech_tokens=" << speechTokens
                  << " prefill_ms=" << prefillMs
                  << " decode_ms=" << decodeMs
                  << " wall_ms=" << wallMs
                  << " tokens_per_second=" << tokensPerSecond
                  << " invalid_outputs=" << invalid
                  << " vm_rss=" << currentRss() << std::endl;
        if (speechTokens == 0 || invalid != 0) {
            return 7;
        }
    }

    return 0;
}

#ifndef COSYVOICE_LLM_LIBRARY
int main(int argc, const char* argv[]) {
    return cosyVoiceLlmMain(argc, argv);
}
#endif
