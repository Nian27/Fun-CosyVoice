#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torchaudio


DEFAULT_TEXT = "窗外的雨渐渐停了，他放下手中的书，认真听着远处传来的钟声。"
DEFAULT_PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呀。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--student-patch", type=Path, required=True)
    parser.add_argument("--prompt-wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--student-steps", type=int, default=10)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def single_branch_solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
    t = t_span[0].unsqueeze(dim=0)
    dt = t_span[1] - t_span[0]
    dtype = spks.dtype
    for step in range(1, len(t_span)):
        t_in = t.repeat(x.size(0)).to(dtype=dtype)
        velocity = self.forward_estimator(
            x.to(dtype=dtype),
            mask.to(dtype=dtype),
            mu.to(dtype=dtype),
            t_in,
            spks.to(dtype=dtype),
            cond.to(dtype=dtype),
            streaming,
        )
        x = x + dt * velocity
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
    return x.float()


def audio_stats(audio: torch.Tensor, sample_rate: int) -> dict:
    values = audio.detach().float().cpu().flatten()
    return {
        "samples": values.numel(),
        "duration_seconds": values.numel() / sample_rate,
        "finite": bool(torch.isfinite(values).all()),
        "peak_abs": float(values.abs().max()),
        "rms": float(torch.sqrt(values.square().mean())),
        "clipped_ratio": float((values.abs() >= 0.999).float().mean()),
    }


def compare_tensor(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference_count = reference.numel()
    candidate_count = candidate.numel()
    count = min(reference_count, candidate_count)
    reference = reference.float().flatten()[:count]
    candidate = candidate.float().flatten()[:count]
    error = candidate - reference
    cosine = torch.nn.functional.cosine_similarity(
        reference.unsqueeze(0), candidate.unsqueeze(0)
    ).item()
    signal = reference.square().sum().clamp_min(1e-12)
    noise = error.square().sum().clamp_min(1e-12)
    return {
        "compared_samples": count,
        "sample_count_equal": reference_count == candidate_count,
        "mae": float(error.abs().mean()),
        "rmse": float(torch.sqrt(error.square().mean())),
        "cosine_similarity": cosine,
        "snr_db": float(10.0 * torch.log10(signal / noise)),
    }


def synthesize(
    cosyvoice,
    args,
    speaker_id: str,
    seed: int,
    captured_mels: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    seed_everything(seed)
    captured_mels.clear()
    torch.cuda.synchronize()
    started = time.perf_counter()
    chunks = list(
        cosyvoice.inference_zero_shot(
            args.text,
            args.prompt_text,
            str(args.prompt_wav),
            zero_shot_spk_id=speaker_id,
            stream=False,
            text_frontend=False,
        )
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not chunks:
        raise RuntimeError("CosyVoice returned no audio")
    audio = torch.cat([item["tts_speech"].detach().cpu() for item in chunks], dim=1)
    if len(captured_mels) != 1:
        raise RuntimeError(f"Expected one non-streaming mel, got {len(captured_mels)}")
    stats = audio_stats(audio, cosyvoice.sample_rate)
    stats["elapsed_seconds"] = elapsed
    stats["rtf"] = elapsed / stats["duration_seconds"]
    return audio, captured_mels[0], stats


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import CosyVoice3

    args.output.mkdir(parents=True, exist_ok=True)
    cosyvoice = CosyVoice3(model_dir=str(args.model), fp16=True)
    speaker_id = "cfg-student-evaluation"
    cosyvoice.add_zero_shot_spk(args.prompt_text, str(args.prompt_wav), speaker_id)
    captured_mels = []
    original_flow_inference = cosyvoice.model.flow.inference

    def capture_flow_mel(*flow_args, **flow_kwargs):
        result = original_flow_inference(*flow_args, **flow_kwargs)
        captured_mels.append(result[0].detach().float().cpu())
        return result

    cosyvoice.model.flow.inference = capture_flow_mel

    # Warm the unchanged full pipeline before timing the teacher.
    synthesize(cosyvoice, args, speaker_id, args.seed, captured_mels)
    teacher_audio, teacher_mel, teacher_stats = synthesize(
        cosyvoice, args, speaker_id, args.seed, captured_mels
    )
    torchaudio.save(
        str(args.output / "teacher-cfg.wav"), teacher_audio, cosyvoice.sample_rate
    )

    payload = torch.load(args.student_patch, map_location="cpu", weights_only=False)
    if payload.get("kind") != "cosyvoice3_cfg_student_patch":
        raise RuntimeError("Unsupported student patch")
    estimator = cosyvoice.model.flow.decoder.estimator
    patch = {
        name: value.to(dtype=estimator.proj_out.weight.dtype)
        for name, value in payload["state_dict_patch"].items()
    }
    incompatible = estimator.load_state_dict(patch, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected student patch keys: {incompatible.unexpected_keys}")
    decoder = cosyvoice.model.flow.decoder
    decoder.solve_euler = MethodType(single_branch_solve_euler, decoder)
    original_decoder_forward = decoder.forward

    def selected_student_steps(*decoder_args, **decoder_kwargs):
        decoder_kwargs["n_timesteps"] = args.student_steps
        return original_decoder_forward(*decoder_args, **decoder_kwargs)

    decoder.forward = selected_student_steps

    # Warm kernels again because the estimator batch changes from two to one.
    synthesize(cosyvoice, args, speaker_id, args.seed, captured_mels)
    student_audio, student_mel, student_stats = synthesize(
        cosyvoice, args, speaker_id, args.seed, captured_mels
    )
    torchaudio.save(
        str(args.output / "student-single-branch.wav"),
        student_audio,
        cosyvoice.sample_rate,
    )

    report = {
        "text": args.text,
        "seed": args.seed,
        "student_patch": str(args.student_patch),
        "teacher_steps": 10,
        "student_steps": args.student_steps,
        "teacher": teacher_stats,
        "student": student_stats,
        "waveform_comparison": compare_tensor(teacher_audio, student_audio),
        "mel_comparison": compare_tensor(teacher_mel, student_mel),
        "notes": [
            "Waveform similarity is diagnostic only and is not a perceptual quality score.",
            "Student step count is reported separately from the ten-step teacher.",
        ],
    }
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not teacher_stats["finite"] or not student_stats["finite"]:
        return 2
    if not math.isfinite(report["mel_comparison"]["rmse"]):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
