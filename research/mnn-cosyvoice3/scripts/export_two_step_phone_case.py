#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml


INPUT_NAMES = ("x", "mask", "mu", "spks", "cond")


def log(stage: str) -> None:
    print(f"[export-phone-case] {stage}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--student-patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--utterance-id", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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
    return flow.decoder.estimator.eval().float()


def find_records(cache: dict, utterance_id: int) -> list[dict]:
    records = [
        record
        for split in ("train", "validation")
        for record in cache[split]
        if int(record["utterance_id"]) == utterance_id
    ]
    if not records:
        raise RuntimeError(f"Utterance {utterance_id} was not found")
    return sorted(records, key=lambda item: int(item["solver_step"]))


def teacher_final(records: list[dict]) -> torch.Tensor:
    last = records[-1]
    last_t = float(last["t"].float().item())
    return last["x"].float() + (1.0 - last_t) * last["target"].float()


def tensor_stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all()),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "rms": float(torch.sqrt(value.square().mean())),
    }


def comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = reference.detach().float().cpu().flatten()
    candidate = candidate.detach().float().cpu().flatten()
    error = candidate - reference
    signal = reference.square().sum().clamp_min(1e-12)
    noise = error.square().sum().clamp_min(1e-12)
    return {
        "mae": float(error.abs().mean()),
        "max_abs": float(error.abs().max()),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                reference.unsqueeze(0), candidate.unsqueeze(0)
            )
        ),
        "snr_db": float(10.0 * torch.log10(signal / noise)),
    }


def write_tensor(output: Path, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().float().contiguous().cpu().numpy().astype("<f4")
    value.tofile(output / f"{name}.bin")


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    log("load_cache_begin")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("kind") != "cosyvoice3_cfg_velocity_cache":
        raise RuntimeError("Unsupported trajectory cache")
    records = find_records(cache, args.utterance_id)
    initial = records[0]
    log("load_cache_done")

    log("load_estimator_begin")
    estimator = load_estimator(args.repo, args.model)
    log("load_estimator_done")
    log("load_student_patch_begin")
    patch_payload = torch.load(
        args.student_patch, map_location="cpu", weights_only=False
    )
    if patch_payload.get("kind") != "cosyvoice3_cfg_student_patch":
        raise RuntimeError("Unsupported student patch")
    incompatible = estimator.load_state_dict(
        {
            name: value.float()
            for name, value in patch_payload["state_dict_patch"].items()
        },
        strict=False,
    )
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected patch keys: {incompatible.unexpected_keys}")
    log("load_student_patch_done")
    log(f"move_estimator_begin device={device}")
    estimator.to(device)
    log("move_estimator_done")

    inputs = {
        name: initial[name].float().to(device)
        for name in INPUT_NAMES
    }
    midpoint = 1.0 - math.cos(math.pi / 4.0)
    x = inputs["x"].clone()
    velocities = []
    for index, (time_point, delta) in enumerate(
        ((0.0, midpoint), (midpoint, 1.0 - midpoint))
    ):
        log(f"student_step_{index}_begin")
        t = torch.tensor([time_point], dtype=torch.float32, device=device)
        velocity = estimator(
            x,
            inputs["mask"],
            inputs["mu"],
            t,
            inputs["spks"],
            inputs["cond"],
        )
        velocities.append(velocity.cpu())
        x = x + delta * velocity
        log(f"student_step_{index}_done")

    teacher = teacher_final(records)
    student = x.cpu()
    valid_length = int(initial["mask"].float().sum().item())
    prompt_activity = initial["cond"].float().abs().sum(dim=1).squeeze(0)
    nonzero_frames = torch.nonzero(prompt_activity > 0, as_tuple=False).flatten()
    prompt_length = int(nonzero_frames[-1].item() + 1) if nonzero_frames.numel() else 0
    target_length = valid_length - prompt_length
    teacher_target = teacher[:, :, prompt_length:valid_length]
    student_target = student[:, :, prompt_length:valid_length]

    args.output.mkdir(parents=True, exist_ok=True)
    log("write_outputs_begin")
    for name in INPUT_NAMES:
        write_tensor(args.output, name, initial[name])
    write_tensor(args.output, "velocity_step0", velocities[0])
    write_tensor(args.output, "velocity_step1", velocities[1])
    write_tensor(args.output, "teacher_final", teacher)
    write_tensor(args.output, "student_final", student)
    write_tensor(args.output, "teacher_target_mel", teacher_target)
    write_tensor(args.output, "student_target_mel", student_target)

    report = {
        "format_version": 1,
        "utterance_id": args.utterance_id,
        "text": initial.get("text", ""),
        "sequence_length": int(initial["x"].shape[2]),
        "valid_length": valid_length,
        "prompt_length": prompt_length,
        "target_length": target_length,
        "target_audio_seconds": target_length * 480 / 24000,
        "time_points": [0.0, midpoint, 1.0],
        "student_vs_teacher_full": comparison(
            teacher[:, :, :valid_length], student[:, :, :valid_length]
        ),
        "student_vs_teacher_target": comparison(teacher_target, student_target),
        "tensors": {
            name: tensor_stats(value)
            for name, value in {
                **{name: initial[name] for name in INPUT_NAMES},
                "velocity_step0": velocities[0],
                "velocity_step1": velocities[1],
                "teacher_final": teacher,
                "student_final": student,
                "teacher_target_mel": teacher_target,
                "student_target_mel": student_target,
            }.items()
        },
    }
    (args.output / "case.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log("write_outputs_done")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
