#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--student-patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    return flow.decoder.estimator.eval()


def group_records(records: list[dict]) -> list[list[dict]]:
    groups = {}
    for record in records:
        groups.setdefault(int(record["utterance_id"]), []).append(record)
    return [
        sorted(items, key=lambda item: int(item["solver_step"]))
        for _, items in sorted(groups.items())
    ]


def teacher_final(records: list[dict]) -> torch.Tensor:
    last = records[-1]
    last_t = float(last["t"].float().item())
    return last["x"].float() + (1.0 - last_t) * last["target"].float()


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = reference.float().flatten()
    candidate = candidate.float().flatten()
    error = candidate - reference
    signal = reference.square().sum().clamp_min(1e-12)
    noise = error.square().sum().clamp_min(1e-12)
    return {
        "elements": reference.numel(),
        "mae": float(error.abs().mean()),
        "rmse": float(torch.sqrt(error.square().mean())),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                reference.unsqueeze(0), candidate.unsqueeze(0)
            )
        ),
        "snr_db": float(10.0 * torch.log10(signal / noise)),
    }


@torch.inference_mode()
def evaluate_trajectory(
    estimator: torch.nn.Module,
    records: list[dict],
    device: torch.device,
) -> dict:
    initial = records[0]
    dtype = torch.float32
    x = initial["x"].to(device=device, dtype=dtype)
    mask = initial["mask"].to(device=device, dtype=dtype)
    mu = initial["mu"].to(device=device, dtype=dtype)
    spks = initial["spks"].to(device=device, dtype=dtype)
    cond = initial["cond"].to(device=device, dtype=dtype)
    midpoint_t = 1.0 - math.cos(math.pi / 4.0)
    time_points = (0.0, midpoint_t)
    deltas = (midpoint_t, 1.0 - midpoint_t)
    for time_point, delta in zip(time_points, deltas):
        t = torch.tensor([time_point], device=device, dtype=dtype)
        x = x + delta * estimator(x, mask, mu, t, spks, cond)

    reference = teacher_final(records)
    candidate = x.cpu()
    valid_length = int(initial["mask"].float().sum().item())
    prompt_activity = initial["cond"].float().abs().sum(dim=1).squeeze(0)
    nonzero_frames = torch.nonzero(prompt_activity > 0, as_tuple=False).flatten()
    prompt_length = int(nonzero_frames[-1].item() + 1) if nonzero_frames.numel() else 0
    target_start = min(prompt_length, valid_length)
    return {
        "utterance_id": int(initial["utterance_id"]),
        "text": initial.get("text", ""),
        "sequence_length": int(initial["x"].shape[2]),
        "valid_length": valid_length,
        "prompt_length": prompt_length,
        "full": metrics(reference[:, :, :valid_length], candidate[:, :, :valid_length]),
        "target_only": metrics(
            reference[:, :, target_start:valid_length],
            candidate[:, :, target_start:valid_length],
        ),
    }


def summarize(results: list[dict], section: str) -> dict:
    keys = ("mae", "rmse", "cosine_similarity", "snr_db")
    return {
        f"mean_{key}": float(np.mean([item[section][key] for item in results]))
        for key in keys
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("kind") != "cosyvoice3_cfg_velocity_cache":
        raise RuntimeError("Unsupported trajectory cache")
    estimator = load_estimator(args.repo, args.model).to(device=device, dtype=torch.float32)
    patch_payload = torch.load(
        args.student_patch, map_location="cpu", weights_only=False
    )
    if patch_payload.get("kind") != "cosyvoice3_cfg_student_patch":
        raise RuntimeError("Unsupported student patch")
    patch = {
        name: value.float() for name, value in patch_payload["state_dict_patch"].items()
    }
    incompatible = estimator.load_state_dict(patch, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected student patch keys: {incompatible.unexpected_keys}")

    results = [
        evaluate_trajectory(estimator, records, device)
        for records in group_records(cache["validation"])
    ]
    report = {
        "cache": str(args.cache),
        "student_patch": str(args.student_patch),
        "validation_utterances": len(results),
        "time_points": [0.0, 1.0 - math.cos(math.pi / 4.0), 1.0],
        "full_summary": summarize(results, "full"),
        "target_only_summary": summarize(results, "target_only"),
        "utterances": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
