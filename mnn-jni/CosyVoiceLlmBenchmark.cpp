#include <llm/llm.hpp>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using MNN::Transformer::Llm;

namespace {

constexpr int kSpeechOffset = 151924;
constexpr int kSpeechTokenMax = kSpeechOffset + 6560;
constexpr int kSpeechEos = kSpeechOffset + 6562;

void printIds(const char* label, const std::vector<int>& ids) {
    std::cout << label << "=[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            std::cout << ',';
        }
        std::cout << ids[i];
    }
    std::cout << "]\n";
}

bool isSpeechOutput(int token) {
    return (token >= kSpeechOffset && token <= kSpeechTokenMax) || token == kSpeechEos;
}

std::vector<int> readPromptSpeechTokens(const std::string& path) {
    std::vector<int> tokens;
    if (path.empty()) {
        return tokens;
    }
    std::ifstream input(path);
    int token = 0;
    char separator = 0;
    while (input >> token) {
        tokens.push_back(token);
        if (!(input >> separator)) {
            break;
        }
        if (separator != ',') {
            break;
        }
    }
    return tokens;
}

}  // namespace

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " config.json text [max_tokens] [prompt_speech_tokens.csv] [output_speech_tokens.csv]\n";
        return 2;
    }

    const std::string configPath = argv[1];
    std::string text = argv[2];
    if (!text.empty() && text.front() == '@') {
        std::ifstream textInput(text.substr(1));
        if (!textInput) {
            std::cerr << "cannot_read_text_file=" << text.substr(1) << '\n';
            return 2;
        }
        text.assign(std::istreambuf_iterator<char>(textInput),
                    std::istreambuf_iterator<char>());
        while (!text.empty() && (text.back() == '\r' || text.back() == '\n')) {
            text.pop_back();
        }
    }
    const int maxTokens = argc >= 4 ? std::max(1, std::atoi(argv[3])) : 30;
    const std::string promptTokenPath = argc >= 5 ? argv[4] : "";
    const std::string outputTokenPath = argc >= 6 ? argv[5] : "";

    std::unique_ptr<Llm> llm(Llm::createLLM(configPath));
    if (!llm || !llm->load()) {
        std::cerr << "load=false\n";
        return 3;
    }

    std::string prompt = llm->apply_chat_template(text);
    if (prompt.empty()) {
        prompt = text;
    }
    auto inputIds = llm->tokenizer_encode(prompt);
    const auto promptSpeechTokens = readPromptSpeechTokens(promptTokenPath);
    for (int token : promptSpeechTokens) {
        inputIds.push_back(token + kSpeechOffset);
    }
    printIds("input_ids", inputIds);
    std::cout << "prompt_speech_tokens=" << promptSpeechTokens.size() << '\n';

    const auto outputIds = llm->generate(inputIds, maxTokens);
    printIds("output_ids", outputIds);

    int validSpeech = 0;
    int invalid = 0;
    std::vector<int> outputSpeechTokens;
    for (int token : outputIds) {
        if (isSpeechOutput(token)) {
            ++validSpeech;
            if (token >= kSpeechOffset && token <= kSpeechTokenMax) {
                outputSpeechTokens.push_back(token - kSpeechOffset);
            }
        } else {
            ++invalid;
        }
    }
    if (!outputTokenPath.empty()) {
        std::ofstream output(outputTokenPath);
        if (!output) {
            std::cerr << "cannot_write_output_speech_tokens=" << outputTokenPath << '\n';
            return 5;
        }
        for (size_t i = 0; i < outputSpeechTokens.size(); ++i) {
            if (i) {
                output << ',';
            }
            output << outputSpeechTokens[i];
        }
        output << '\n';
        std::cout << "output_speech_token_file=" << outputTokenPath << '\n';
    }

    const auto* context = llm->getContext();
    const double prefillSeconds = context->prefill_us / 1000000.0;
    const double decodeSeconds = context->decode_us / 1000000.0;
    std::cout << std::fixed << std::setprecision(6)
              << "prompt_tokens=" << context->prompt_len << '\n'
              << "generated_tokens=" << context->gen_seq_len << '\n'
              << "valid_speech_tokens=" << validSpeech << '\n'
              << "output_speech_tokens=" << outputSpeechTokens.size() << '\n'
              << "invalid_tokens=" << invalid << '\n'
              << "stopped_by_speech_eos="
              << (!outputIds.empty() && outputIds.back() == kSpeechEos ? "true" : "false") << '\n'
              << "prefill_seconds=" << prefillSeconds << '\n'
              << "decode_seconds=" << decodeSeconds << '\n'
              << "decode_tokens_per_second="
              << (decodeSeconds > 0.0 ? context->gen_seq_len / decodeSeconds : 0.0) << '\n'
              << "status=" << static_cast<int>(context->status) << '\n';
    return invalid == 0 && !outputIds.empty() ? 0 : 4;
}
