#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re

import numpy as np
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
from opencc import OpenCC


T2S = OpenCC("t2s")


def load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    audio, source_rate = torchaudio.load(str(path))
    audio = audio.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio


def speaker_embedding(session: onnxruntime.InferenceSession, path: Path) -> np.ndarray:
    audio = load_mono(path, 16000)
    feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    name = session.get_inputs()[0].name
    embedding = session.run(None, {name: feat.unsqueeze(0).numpy()})[0].reshape(-1)
    norm = np.linalg.norm(embedding)
    return embedding / max(norm, 1e-12)


def normalize_text(text: str) -> str:
    simplified = T2S.convert(text)
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", simplified)).lower()


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--campplus", type=Path, required=True)
    parser.add_argument("--prompt-wav", type=Path, required=True)
    parser.add_argument("--expected-text", required=True)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--whisper-cache", type=Path, required=True)
    args = parser.parse_args()

    paths = {steps: args.audio_dir / f"flow-{steps}-steps.wav" for steps in (10, 4, 2)}
    for path in [*paths.values(), args.campplus, args.prompt_wav]:
        if not path.is_file():
            raise FileNotFoundError(path)

    session = onnxruntime.InferenceSession(
        str(args.campplus), providers=["CPUExecutionProvider"]
    )
    embeddings = {steps: speaker_embedding(session, path) for steps, path in paths.items()}
    prompt_embedding = speaker_embedding(session, args.prompt_wav)

    args.whisper_cache.mkdir(parents=True, exist_ok=True)
    asr = whisper.load_model(
        args.whisper_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        download_root=str(args.whisper_cache),
    )
    expected = normalize_text(args.expected_text)
    results = []
    for steps, path in paths.items():
        audio = load_mono(path, 16000).squeeze(0).numpy()
        transcript = asr.transcribe(
            audio,
            language="zh",
            task="transcribe",
            temperature=0,
            fp16=torch.cuda.is_available(),
        )["text"].strip()
        normalized = normalize_text(transcript)
        distance = edit_distance(expected, normalized)
        results.append(
            {
                "steps": steps,
                "transcript": transcript,
                "normalized_transcript": normalized,
                "character_errors": distance,
                "cer": distance / max(len(expected), 1),
                "speaker_cosine_vs_10": float(np.dot(embeddings[steps], embeddings[10])),
                "speaker_cosine_vs_prompt": float(np.dot(embeddings[steps], prompt_embedding)),
            }
        )

    report = {
        "expected_text": args.expected_text,
        "normalized_expected_text": expected,
        "whisper_model": args.whisper_model,
        "results": results,
        "note": "Whisper CER is a relative screening metric, not a substitute for blind listening.",
    }
    target = args.audio_dir / "quality-analysis.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
