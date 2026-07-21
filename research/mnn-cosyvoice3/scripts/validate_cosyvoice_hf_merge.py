#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import torch
from hyperpyyaml import load_hyperpyyaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "third_party" / "Matcha-TTS"))

    metadata = json.loads(
        (args.merged / "cosyvoice3_metadata.json").read_text(encoding="utf-8")
    )
    with (args.model / "cosyvoice3.yaml").open("r", encoding="utf-8") as stream:
        configs = load_hyperpyyaml(
            stream,
            overrides={"qwen_pretrain_path": str(args.model / "CosyVoice-BlankEN")},
        )
    original = configs["llm"]
    original.load_state_dict(
        torch.load(args.model / "llm.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    original.eval()

    merged = AutoModelForCausalLM.from_pretrained(
        args.merged,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.merged, local_files_only=True)

    text = (
        "You are a helpful assistant.<|endofprompt|>"
        "今天阳光很好，我们一起去公园散步。"
    )
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    prompt_speech = [23, 418, 2048, 6559]
    speech_offset = int(metadata["speech_token_offset"])
    sos = int(metadata["base_speech_token_size"])
    task_id = sos + 2

    original_text = torch.tensor([text_ids], dtype=torch.long)
    original_prompt = torch.tensor([prompt_speech], dtype=torch.long)
    with torch.inference_mode():
        text_embeds = original.llm.model.model.embed_tokens(original_text)
        lm_input = torch.cat(
            [
                original.speech_embedding.weight[sos].reshape(1, 1, -1),
                text_embeds,
                original.speech_embedding.weight[task_id].reshape(1, 1, -1),
                original.speech_embedding(original_prompt),
            ],
            dim=1,
        )
        original_output = original.llm.model(
            inputs_embeds=lm_input,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        original_logits = original.llm_decoder(original_output.hidden_states[-1][:, -1])

        merged_ids = torch.tensor(
            [
                [speech_offset + sos]
                + text_ids
                + [speech_offset + task_id]
                + [speech_offset + value for value in prompt_speech]
            ],
            dtype=torch.long,
        )
        merged_output = merged(merged_ids, use_cache=False, return_dict=True)
        merged_logits = merged_output.logits[
            :, -1, speech_offset : speech_offset + original_logits.shape[-1]
        ].float()

    reference = original_logits.float()
    difference = (reference - merged_logits).abs()
    reference_top = int(reference.argmax(dim=-1).item())
    merged_top = int(merged_logits.argmax(dim=-1).item())
    report = {
        "text_token_count": len(text_ids),
        "prompt_speech_token_count": len(prompt_speech),
        "speech_token_offset": speech_offset,
        "original_top_token": reference_top,
        "merged_top_token": merged_top,
        "argmax_equal": reference_top == merged_top,
        "max_abs_logit_error": float(difference.max().item()),
        "mean_abs_logit_error": float(difference.mean().item()),
        "reference_rms": float(torch.sqrt(reference.square().mean()).item()),
        "merged_rms": float(torch.sqrt(merged_logits.square().mean()).item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["argmax_equal"] or report["max_abs_logit_error"] > 0.1:
        raise RuntimeError("Merged HuggingFace LLM failed the logit equivalence gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
