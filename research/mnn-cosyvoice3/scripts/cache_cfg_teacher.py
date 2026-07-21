#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--cfg-rate", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_estimator(repo: Path, model: Path) -> torch.nn.Module:
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "third_party" / "Matcha-TTS"))
    with (model / "cosyvoice3.yaml").open("r", encoding="utf-8") as stream:
        configs = load_hyperpyyaml(
            stream,
            overrides={"qwen_pretrain_path": str(model / "CosyVoice-BlankEN")},
        )
    flow = configs["flow"]
    flow.load_state_dict(
        torch.load(model / "flow.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    return flow.decoder.estimator.eval()


def make_sample(
    generator: torch.Generator,
    sequence_length: int,
    channels: int,
) -> dict[str, torch.Tensor]:
    shape = (1, channels, sequence_length)
    valid_length = int(
        torch.randint(
            max(8, sequence_length * 3 // 4),
            sequence_length + 1,
            (1,),
            generator=generator,
        ).item()
    )
    prompt_length = int(
        torch.randint(0, max(1, valid_length * 3 // 10 + 1), (1,), generator=generator).item()
    )
    x = torch.randn(shape, generator=generator)
    mu = torch.randn(shape, generator=generator) * 0.75
    spks = torch.randn((1, channels), generator=generator)
    cond = torch.zeros(shape)
    if prompt_length:
        cond[:, :, :prompt_length] = torch.randn(
            (1, channels, prompt_length), generator=generator
        ) * 0.5
    mask = torch.zeros((1, 1, sequence_length))
    mask[:, :, :valid_length] = 1.0
    t = torch.rand((1,), generator=generator) * 0.96 + 0.02
    return {
        "x": x,
        "mask": mask,
        "mu": mu,
        "t": t,
        "spks": spks,
        "cond": cond,
        "valid_length": torch.tensor(valid_length),
        "prompt_length": torch.tensor(prompt_length),
    }


def main() -> int:
    args = parse_args()
    if args.sample_count <= args.validation_count:
        raise ValueError("sample-count must be greater than validation-count")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    seed_everything(args.seed)
    device = torch.device(args.device)
    compute_dtype = getattr(torch, args.compute_dtype)
    storage_dtype = getattr(torch, args.storage_dtype)
    estimator = load_estimator(args.repo, args.model).to(device=device, dtype=compute_dtype)
    channels = estimator.out_channels
    generator = torch.Generator().manual_seed(args.seed)
    cached = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for index in range(args.sample_count):
            sample = make_sample(generator, args.sequence_length, channels)
            conditional_names = ("x", "mask", "mu", "t", "spks", "cond")
            conditional = {
                name: sample[name].to(device=device, dtype=compute_dtype)
                for name in conditional_names
            }
            teacher_inputs = {
                "x": conditional["x"].repeat(2, 1, 1),
                "mask": conditional["mask"].repeat(2, 1, 1),
                "mu": torch.cat((conditional["mu"], torch.zeros_like(conditional["mu"]))),
                "t": conditional["t"].repeat(2),
                "spks": torch.cat(
                    (conditional["spks"], torch.zeros_like(conditional["spks"]))
                ),
                "cond": torch.cat(
                    (conditional["cond"], torch.zeros_like(conditional["cond"]))
                ),
            }
            teacher_output = estimator(**teacher_inputs)
            velocity_cond, velocity_uncond = teacher_output.chunk(2, dim=0)
            velocity_guided = (
                (1.0 + args.cfg_rate) * velocity_cond
                - args.cfg_rate * velocity_uncond
            )
            record = {
                name: conditional[name].cpu().to(storage_dtype)
                for name in conditional_names
            }
            record.update(
                {
                    "target": velocity_guided.cpu().to(storage_dtype),
                    "teacher_cond": velocity_cond.cpu().to(storage_dtype),
                    "valid_length": sample["valid_length"],
                    "prompt_length": sample["prompt_length"],
                }
            )
            cached.append(record)
            print(f"cached {index + 1}/{args.sample_count}", flush=True)

    split = args.sample_count - args.validation_count
    payload = {
        "format_version": 1,
        "kind": "cosyvoice3_cfg_velocity_cache",
        "synthetic_inputs": True,
        "seed": args.seed,
        "cfg_rate": args.cfg_rate,
        "sequence_length": args.sequence_length,
        "storage_dtype": args.storage_dtype,
        "train": cached[:split],
        "validation": cached[split:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report = {
        "cache": str(args.output),
        "bytes": args.output.stat().st_size,
        "sample_count": args.sample_count,
        "train_count": split,
        "validation_count": args.validation_count,
        "sequence_length": args.sequence_length,
        "cfg_rate": args.cfg_rate,
        "synthetic_inputs": True,
        "compute_dtype": args.compute_dtype,
        "storage_dtype": args.storage_dtype,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "target_rms": float(
            torch.sqrt(torch.cat([item["target"].float().flatten() for item in cached]).square().mean())
        ),
        "cfg_delta_rms": float(
            torch.sqrt(
                torch.cat(
                    [
                        (item["target"].float() - item["teacher_cond"].float()).flatten()
                        for item in cached
                    ]
                ).square().mean()
            )
        ),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
