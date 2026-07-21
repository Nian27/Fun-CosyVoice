#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


DEFAULT_PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呀。"
)
DEFAULT_TEXTS = [
    "今天阳光很好，我们一起去公园散步，顺便买一些新鲜水果。",
    "雨停以后，街道上的灯光映在积水里，远处传来缓慢的脚步声。",
    "他停顿片刻，低声说道：这件事还没有结束，我们必须继续调查。",
    "请把桌上的书递给我，然后关上窗户，外面的风越来越大了。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", action="append", dest="texts")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--validation-utterances", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CfgCapture(nn.Module):
    def __init__(self, estimator: nn.Module, cfg_rate: float, storage_dtype: torch.dtype):
        super().__init__()
        self.estimator = estimator
        self.cfg_rate = cfg_rate
        self.storage_dtype = storage_dtype
        self.records = []
        self.utterance_id = -1
        self.text = ""
        self.step = 0

    def start_utterance(self, utterance_id: int, text: str) -> None:
        self.utterance_id = utterance_id
        self.text = text
        self.step = 0

    def forward(self, x, mask, mu, t, spks, cond, streaming=False):
        output = self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        if output.shape[0] != 2:
            raise RuntimeError(f"Expected CFG batch 2, got {output.shape[0]}")
        velocity_cond, velocity_uncond = output.chunk(2, dim=0)
        target = (
            (1.0 + self.cfg_rate) * velocity_cond
            - self.cfg_rate * velocity_uncond
        )
        record = {
            "x": x[:1].detach().cpu().to(self.storage_dtype),
            "mask": mask[:1].detach().cpu().to(self.storage_dtype),
            "mu": mu[:1].detach().cpu().to(self.storage_dtype),
            "t": t[:1].detach().cpu().to(self.storage_dtype),
            "spks": spks[:1].detach().cpu().to(self.storage_dtype),
            "cond": cond[:1].detach().cpu().to(self.storage_dtype),
            "target": target.detach().cpu().to(self.storage_dtype),
            "teacher_cond": velocity_cond.detach().cpu().to(self.storage_dtype),
            "utterance_id": self.utterance_id,
            "solver_step": self.step,
            "text": self.text,
        }
        self.records.append(record)
        self.step += 1
        return output


def main() -> int:
    args = parse_args()
    if args.text_file is not None and args.texts:
        raise ValueError("Use either --text-file or repeated --text, not both")
    if args.text_file is not None:
        texts = [
            line.strip()
            for line in args.text_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        texts = args.texts or DEFAULT_TEXTS
    if len(texts) < 2:
        raise ValueError("At least two utterances are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real trajectory capture")
    if args.validation_utterances < 1 or args.validation_utterances >= len(texts):
        raise ValueError("validation-utterances must leave at least one training utterance")

    seed_everything(args.seed)
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import CosyVoice3

    storage_dtype = getattr(torch, args.storage_dtype)
    load_started = time.perf_counter()
    cosyvoice = CosyVoice3(model_dir=str(args.model), fp16=True)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    decoder = cosyvoice.model.flow.decoder
    capture = CfgCapture(
        decoder.estimator,
        cfg_rate=float(decoder.inference_cfg_rate),
        storage_dtype=storage_dtype,
    )
    decoder.estimator = capture
    speaker_id = "cfg-distill-capture"
    cosyvoice.add_zero_shot_spk(args.prompt_text, str(args.prompt_wav), speaker_id)

    utterances = []
    for utterance_id, text in enumerate(texts):
        seed_everything(args.seed + utterance_id)
        capture.start_utterance(utterance_id, text)
        started = time.perf_counter()
        chunks = list(
            cosyvoice.inference_zero_shot(
                text,
                args.prompt_text,
                str(args.prompt_wav),
                zero_shot_spk_id=speaker_id,
                stream=False,
                text_frontend=False,
            )
        )
        torch.cuda.synchronize()
        if not chunks:
            raise RuntimeError(f"CosyVoice returned no audio for utterance {utterance_id}")
        audio_samples = sum(chunk["tts_speech"].numel() for chunk in chunks)
        utterances.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "trajectory_records": capture.step,
                "audio_samples": audio_samples,
                "audio_seconds": audio_samples / cosyvoice.sample_rate,
                "capture_seconds": time.perf_counter() - started,
            }
        )
        print(json.dumps(utterances[-1], ensure_ascii=False), flush=True)

    validation_start = len(texts) - args.validation_utterances
    train = [item for item in capture.records if item["utterance_id"] < validation_start]
    validation = [
        item for item in capture.records if item["utterance_id"] >= validation_start
    ]
    payload = {
        "format_version": 1,
        "kind": "cosyvoice3_cfg_velocity_cache",
        "synthetic_inputs": False,
        "seed": args.seed,
        "cfg_rate": capture.cfg_rate,
        "storage_dtype": args.storage_dtype,
        "prompt_wav": str(args.prompt_wav),
        "prompt_text": args.prompt_text,
        "utterances": utterances,
        "train": train,
        "validation": validation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    all_records = train + validation
    report = {
        "cache": str(args.output),
        "bytes": args.output.stat().st_size,
        "synthetic_inputs": False,
        "cfg_rate": capture.cfg_rate,
        "load_seconds": load_seconds,
        "utterance_count": len(texts),
        "train_utterances": validation_start,
        "validation_utterances": args.validation_utterances,
        "train_records": len(train),
        "validation_records": len(validation),
        "sequence_lengths": sorted({int(item["x"].shape[2]) for item in all_records}),
        "target_rms": float(
            torch.sqrt(
                torch.cat([item["target"].float().flatten() for item in all_records])
                .square()
                .mean()
            )
        ),
        "cfg_delta_rms": float(
            torch.sqrt(
                torch.cat(
                    [
                        (item["target"].float() - item["teacher_cond"].float()).flatten()
                        for item in all_records
                    ]
                )
                .square()
                .mean()
            )
        ),
        "utterances": utterances,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
