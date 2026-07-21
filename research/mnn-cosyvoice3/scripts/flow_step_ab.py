#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F


DEFAULT_TEXT = "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"
DEFAULT_PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def audio_stats(audio: torch.Tensor, sample_rate: int, elapsed: float) -> dict:
    audio = audio.detach().float().cpu().flatten()
    duration = audio.numel() / sample_rate
    finite = torch.isfinite(audio)
    finite_audio = audio[finite]
    peak = finite_audio.abs().max().item() if finite_audio.numel() else math.nan
    rms = torch.sqrt(torch.mean(finite_audio.square())).item() if finite_audio.numel() else math.nan
    return {
        "duration_seconds": duration,
        "elapsed_seconds": elapsed,
        "rtf": elapsed / duration if duration else math.inf,
        "finite_ratio": finite.float().mean().item(),
        "peak_abs": peak,
        "rms": rms,
        "clipped_ratio": (finite_audio.abs() >= 0.999).float().mean().item()
        if finite_audio.numel()
        else math.nan,
    }


def mel_distance(reference: torch.Tensor, candidate: torch.Tensor, sample_rate: int) -> dict:
    length = min(reference.shape[-1], candidate.shape[-1])
    reference = reference[..., :length].float().cpu()
    candidate = candidate[..., :length].float().cpu()
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        power=2.0,
    )
    ref_log = torch.log(mel(reference).clamp_min(1e-7))
    cand_log = torch.log(mel(candidate).clamp_min(1e-7))
    return {
        "log_mel_l1_vs_10": F.l1_loss(cand_log, ref_log).item(),
        "waveform_rmse_vs_10": torch.sqrt(F.mse_loss(candidate, reference)).item(),
        "compared_samples": length,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 4, 2])
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import CosyVoice3

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this desktop quality gate")
    for required in ("llm.pt", "flow.pt", "hift.pt", "speech_tokenizer_v3.onnx"):
        if not (args.model / required).is_file():
            raise FileNotFoundError(args.model / required)

    args.output.mkdir(parents=True, exist_ok=True)
    model = CosyVoice3(model_dir=str(args.model), fp16=True)
    decoder = model.model.flow.decoder
    original_forward = decoder.forward
    active_steps = args.steps[0]

    def forward_with_selected_steps(*forward_args, **forward_kwargs):
        forward_kwargs["n_timesteps"] = active_steps
        return original_forward(*forward_args, **forward_kwargs)

    decoder.forward = forward_with_selected_steps
    results = []
    generated = {}

    for steps in args.steps:
        active_steps = steps
        seed_everything(args.seed)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        chunks = list(
            model.inference_zero_shot(
                args.text,
                args.prompt_text,
                str(args.prompt_wav),
                stream=False,
                text_frontend=False,
            )
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not chunks:
            raise RuntimeError(f"No audio returned for {steps} steps")
        audio = torch.cat([chunk["tts_speech"].detach().cpu() for chunk in chunks], dim=1)
        path = args.output / f"flow-{steps}-steps.wav"
        torchaudio.save(str(path), audio, model.sample_rate)
        stats = audio_stats(audio, model.sample_rate, elapsed)
        stats.update(
            {
                "steps": steps,
                "wav": str(path),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        generated[steps] = audio
        results.append(stats)
        print(json.dumps(stats, ensure_ascii=False), flush=True)

    reference = generated[10]
    for item in results:
        if item["steps"] == 10:
            item.update(
                {"log_mel_l1_vs_10": 0.0, "waveform_rmse_vs_10": 0.0, "compared_samples": reference.shape[-1]}
            )
        else:
            item.update(mel_distance(reference, generated[item["steps"]], model.sample_rate))

    report = {
        "model": str(args.model),
        "repo": str(args.repo),
        "text": args.text,
        "prompt_text": args.prompt_text,
        "prompt_wav": str(args.prompt_wav),
        "seed": args.seed,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "results": results,
        "note": "Objective distances are screening metrics; final quality requires blind listening and ASR/speaker tests.",
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
