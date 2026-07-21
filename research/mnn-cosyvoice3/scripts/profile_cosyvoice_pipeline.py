#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import random
import sys
import threading
import time

import numpy as np
import torch
import torchaudio


DEFAULT_TEXT = "今天阳光很好，我们一起去公园散步，顺便买一些新鲜水果。"
DEFAULT_PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呀。"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timings:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self.values = {
                "frontend_seconds": [],
                "llm_seconds": [],
                "flow_seconds": [],
                "hift_seconds": [],
                "empty_cache_seconds": [],
            }

    def add(self, name: str, elapsed: float) -> None:
        with self._lock:
            self.values[name].append(elapsed)

    def totals(self) -> dict:
        with self._lock:
            return {name: sum(values) for name, values in self.values.items()}


def install_timers(cosyvoice, timings: Timings):
    originals = {}

    def wrap(obj, method_name: str, timing_name: str):
        original = getattr(obj, method_name)
        originals[(id(obj), method_name)] = (obj, original)

        def timed(*args, **kwargs):
            sync_cuda()
            started = time.perf_counter()
            result = original(*args, **kwargs)
            sync_cuda()
            timings.add(timing_name, time.perf_counter() - started)
            return result

        setattr(obj, method_name, timed)

    wrap(cosyvoice.frontend, "frontend_zero_shot", "frontend_seconds")
    wrap(cosyvoice.model, "llm_job", "llm_seconds")
    wrap(cosyvoice.model.flow, "inference", "flow_seconds")
    wrap(cosyvoice.model.hift, "inference", "hift_seconds")

    original_empty_cache = torch.cuda.empty_cache

    def timed_empty_cache():
        sync_cuda()
        started = time.perf_counter()
        result = original_empty_cache()
        sync_cuda()
        timings.add("empty_cache_seconds", time.perf_counter() - started)
        return result

    torch.cuda.empty_cache = timed_empty_cache

    def restore() -> None:
        for obj, original in originals.values():
            method_name = next(
                key[1]
                for key, value in originals.items()
                if value[0] is obj and value[1] is original
            )
            setattr(obj, method_name, original)
        torch.cuda.empty_cache = original_empty_cache

    return restore, original_empty_cache, timed_empty_cache


def install_flow_steps(cosyvoice, steps: int):
    decoder = cosyvoice.model.flow.decoder
    original = decoder.forward

    def selected_steps(*args, **kwargs):
        kwargs["n_timesteps"] = steps
        return original(*args, **kwargs)

    decoder.forward = selected_steps
    return lambda: setattr(decoder, "forward", original)


def audio_stats(audio: torch.Tensor, sample_rate: int) -> dict:
    audio = audio.detach().float().cpu().flatten()
    duration = audio.numel() / sample_rate
    finite = torch.isfinite(audio)
    values = audio[finite]
    return {
        "duration_seconds": duration,
        "finite_ratio": finite.float().mean().item(),
        "peak_abs": values.abs().max().item() if values.numel() else math.nan,
        "rms": torch.sqrt(values.square().mean()).item() if values.numel() else math.nan,
        "clipped_ratio": (values.abs() >= 0.999).float().mean().item()
        if values.numel()
        else math.nan,
    }


def synthesize(cosyvoice, args, speaker_id: str, timings: Timings, seed: int) -> tuple:
    seed_everything(seed)
    timings.reset()
    torch.cuda.reset_peak_memory_stats()
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
    sync_cuda()
    total = time.perf_counter() - started
    if not chunks:
        raise RuntimeError("CosyVoice returned no audio")
    audio = torch.cat([chunk["tts_speech"].detach().cpu() for chunk in chunks], dim=1)
    result = audio_stats(audio, cosyvoice.sample_rate)
    result.update(timings.totals())
    result.update(
        {
            "total_seconds": total,
            "rtf": total / result["duration_seconds"],
            "unattributed_seconds": total
            - result["frontend_seconds"]
            - result["llm_seconds"]
            - result["flow_seconds"]
            - result["hift_seconds"]
            - result["empty_cache_seconds"],
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    )
    return audio, result


def median(values):
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def summarize(runs: list) -> dict:
    numeric_keys = [
        "duration_seconds",
        "total_seconds",
        "rtf",
        "frontend_seconds",
        "llm_seconds",
        "flow_seconds",
        "hift_seconds",
        "empty_cache_seconds",
        "unattributed_seconds",
        "peak_abs",
        "rms",
        "clipped_ratio",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    ]
    return {f"median_{key}": median([run[key] for run in runs]) for key in numeric_keys}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import CosyVoice3

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    load_started = time.perf_counter()
    cosyvoice = CosyVoice3(model_dir=str(args.model), fp16=True)
    sync_cuda()
    model_load_seconds = time.perf_counter() - load_started

    timings = Timings()
    restore_timers, native_empty_cache, timed_empty_cache = install_timers(cosyvoice, timings)
    restore_steps = install_flow_steps(cosyvoice, args.steps)

    cache_started = time.perf_counter()
    cosyvoice.add_zero_shot_spk(args.prompt_text, str(args.prompt_wav), "profile-speaker")
    sync_cuda()
    prompt_cache_build_seconds = time.perf_counter() - cache_started

    modes = [
        ("baseline", "", True),
        ("cached_prompt", "profile-speaker", True),
        ("cached_prompt_retain_allocator", "profile-speaker", False),
    ]
    report_modes = []
    try:
        # One global warm-up isolates model/kernel initialization from measured runs.
        torch.cuda.empty_cache = lambda: None
        synthesize(cosyvoice, args, "profile-speaker", timings, args.seed)

        for mode_name, speaker_id, clear_cache in modes:
            torch.cuda.empty_cache = timed_empty_cache if clear_cache else (lambda: None)
            runs = []
            saved_audio = None
            for index in range(args.repeats):
                saved_audio, result = synthesize(
                    cosyvoice, args, speaker_id, timings, args.seed + index
                )
                result["run"] = index + 1
                runs.append(result)
                print(json.dumps({"mode": mode_name, **result}, ensure_ascii=False), flush=True)
            wav_path = args.output / f"{mode_name}-{args.steps}-steps.wav"
            torchaudio.save(str(wav_path), saved_audio, cosyvoice.sample_rate)
            report_modes.append(
                {
                    "mode": mode_name,
                    "speaker_cache": bool(speaker_id),
                    "empty_cache_after_sentence": clear_cache,
                    "wav": str(wav_path),
                    "runs": runs,
                    "summary": summarize(runs),
                }
            )
    finally:
        restore_steps()
        restore_timers()

    report = {
        "repo": str(args.repo),
        "model": str(args.model),
        "prompt_wav": str(args.prompt_wav),
        "text": args.text,
        "steps": args.steps,
        "repeats": args.repeats,
        "model_load_seconds": model_load_seconds,
        "prompt_cache_build_seconds": prompt_cache_build_seconds,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "modes": report_modes,
    }
    report_path = args.output / "pipeline-profile.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
