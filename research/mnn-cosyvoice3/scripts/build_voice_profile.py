#!/usr/bin/env python3
import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


MODEL_ID = "Fun-CosyVoice3-0.5B-2512-RL-distilled-v1"
SYSTEM_PROMPT = "You are a helpful assistant.<|endofprompt|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Fun阅读 CosyVoice voice profile ZIP")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(parts: list[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not args.profile_id.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("profile-id 只能包含字母、数字、点、下划线和连字符")
    if args.start < 0 or args.end <= args.start:
        raise ValueError("音频起止时间不正确")

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    start_sample = round(args.start * sample_rate)
    end_sample = min(round(args.end * sample_rate), audio.shape[0])
    segment = audio[start_sample:end_sample]
    if segment.size < sample_rate:
        raise ValueError("提示音频必须至少 1 秒")

    work = args.work_dir / args.profile_id
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    prompt_wav = work / "source.wav"
    sf.write(prompt_wav, segment, sample_rate, subtype="PCM_16")

    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import CosyVoice3

    cosyvoice = CosyVoice3(str(args.model), load_trt=False, load_vllm=False, fp16=True)
    normalized_prompt = cosyvoice.frontend.text_normalize(
        args.prompt_text, split=False, text_frontend=True
    )
    model_input = cosyvoice.frontend.frontend_zero_shot(
        "这是新音色注册校验。",
        normalized_prompt,
        str(prompt_wav),
        cosyvoice.sample_rate,
        "",
    )

    prompt_tokens = (
        model_input["llm_prompt_speech_token"].detach().cpu().reshape(-1).numpy().astype("<i4")
    )
    if not 1 <= prompt_tokens.size <= 125:
        raise ValueError(f"提示语音 Token 数量 {prompt_tokens.size} 超出 1..125，请缩短片段")
    prompt_feat = model_input["prompt_speech_feat"].detach().float().cpu()
    prompt_frames = int(prompt_feat.shape[1])
    if prompt_frames != prompt_tokens.size * 2:
        raise ValueError(
            f"提示 Mel 帧 {prompt_frames} 与语音 Token {prompt_tokens.size} 不满足 2:1"
        )

    embedding = model_input["flow_embedding"].to(cosyvoice.model.device)
    with torch.inference_mode():
        flow_speaker = cosyvoice.model.flow.spk_embed_affine_layer(
            F.normalize(embedding, dim=1)
        ).detach().float().cpu()
    if tuple(flow_speaker.shape) != (1, 80):
        raise ValueError(f"Flow speaker shape 不正确：{tuple(flow_speaker.shape)}")

    token_file = work / "prompt-speech-tokens.csv"
    condition_file = work / "prompt-cond.bin"
    speaker_file = work / "spks.bin"
    token_file.write_text(
        ",".join(str(int(token)) for token in prompt_tokens), encoding="ascii"
    )
    prompt_feat.transpose(1, 2).contiguous().numpy().astype("<f4").tofile(condition_file)
    flow_speaker.contiguous().numpy().astype("<f4").tofile(speaker_file)

    prompt_prefix = SYSTEM_PROMPT + normalized_prompt
    profile_hash = sha256_bytes(
        [
            MODEL_ID.encode("utf-8"),
            prompt_prefix.encode("utf-8"),
            token_file.read_bytes(),
            condition_file.read_bytes(),
            speaker_file.read_bytes(),
        ]
    )
    metadata = {
        "schemaVersion": 1,
        "id": args.profile_id,
        "displayName": args.display_name,
        "modelId": MODEL_ID,
        "promptPrefix": prompt_prefix,
        "promptTokenCount": int(prompt_tokens.size),
        "promptFrameCount": prompt_frames,
        "profileHash": profile_hash,
        "builtIn": False,
        "createdAt": int(time.time() * 1000),
        "source": {
            "fileName": args.audio.name,
            "startSeconds": args.start,
            "endSeconds": args.end,
            "promptText": normalized_prompt,
            "sampleRate": sample_rate,
        },
    }
    metadata_file = work / "profile.json"
    metadata_file.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (metadata_file, token_file, condition_file, speaker_file, prompt_wav):
            archive.write(path, path.name)

    report = {
        "output": str(args.output),
        "profileId": args.profile_id,
        "displayName": args.display_name,
        "promptText": normalized_prompt,
        "segmentSeconds": (end_sample - start_sample) / sample_rate,
        "promptTokens": int(prompt_tokens.size),
        "promptFrames": prompt_frames,
        "profileHash": profile_hash,
        "zipBytes": args.output.stat().st_size,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
