import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-wav", type=Path, required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "third_party" / "Matcha-TTS"))

    from cosyvoice.cli.cosyvoice import CosyVoice3

    cosyvoice = CosyVoice3(str(args.model), load_trt=False, load_vllm=False, fp16=True)
    normalized_prompt = cosyvoice.frontend.text_normalize(
        args.prompt_text, split=False, text_frontend=False
    )
    normalized_text = cosyvoice.frontend.text_normalize(
        args.text, split=False, text_frontend=False
    )
    model_input = cosyvoice.frontend.frontend_zero_shot(
        normalized_text,
        normalized_prompt,
        str(args.prompt_wav),
        cosyvoice.sample_rate,
        "",
    )
    prompt_tokens = (
        model_input["llm_prompt_speech_token"].detach().cpu().reshape(-1).tolist()
    )

    args.tokens.parent.mkdir(parents=True, exist_ok=True)
    args.tokens.write_text(",".join(str(token) for token in prompt_tokens), encoding="ascii")
    args.metadata.write_text(
        json.dumps(
            {
                "prompt_wav": str(args.prompt_wav),
                "prompt_text": normalized_prompt,
                "text": normalized_text,
                "prompt_speech_token_count": len(prompt_tokens),
                "prompt_speech_token_min": min(prompt_tokens),
                "prompt_speech_token_max": max(prompt_tokens),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.metadata.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
