#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    return parser.parse_args()


def group_by_utterance(records: list[dict]) -> list[list[dict]]:
    groups = {}
    for record in records:
        utterance_id = int(record["utterance_id"])
        groups.setdefault(utterance_id, []).append(record)
    return [
        sorted(records, key=lambda item: int(item["solver_step"]))
        for _, records in sorted(groups.items())
    ]


def interpolate_state(records: list[dict], target_t: float) -> torch.Tensor:
    points = [(float(item["t"].float().item()), item["x"].float()) for item in records]
    final_t = 1.0
    last = records[-1]
    last_t = float(last["t"].float().item())
    final_x = last["x"].float() + (final_t - last_t) * last["target"].float()
    points.append((final_t, final_x))
    for (left_t, left_x), (right_t, right_x) in zip(points, points[1:]):
        if left_t <= target_t <= right_t:
            ratio = (target_t - left_t) / max(right_t - left_t, 1e-8)
            return left_x + ratio * (right_x - left_x)
    raise RuntimeError(f"Could not interpolate trajectory at t={target_t}")


def build_macro_records(records: list[dict], storage_dtype: torch.dtype) -> list[dict]:
    if len(records) < 2:
        raise RuntimeError("A teacher trajectory must contain at least two records")
    midpoint_t = 1.0 - math.cos(math.pi / 4.0)
    initial = records[0]
    initial_x = initial["x"].float()
    midpoint_x = interpolate_state(records, midpoint_t)
    final_x = interpolate_state(records, 1.0)
    shared = {
        name: initial[name].float().to(storage_dtype)
        for name in ("mask", "mu", "spks", "cond")
    }
    metadata = {
        "utterance_id": int(initial["utterance_id"]),
        "text": initial.get("text", ""),
    }
    first = {
        "x": initial_x.to(storage_dtype),
        "t": torch.tensor([0.0], dtype=storage_dtype),
        "target": ((midpoint_x - initial_x) / midpoint_t).to(storage_dtype),
        "macro_step": 0,
        **shared,
        **metadata,
    }
    second = {
        "x": midpoint_x.to(storage_dtype),
        "t": torch.tensor([midpoint_t], dtype=storage_dtype),
        "target": ((final_x - midpoint_x) / (1.0 - midpoint_t)).to(storage_dtype),
        "macro_step": 1,
        **shared,
        **metadata,
    }
    return [first, second]


def convert_split(records: list[dict], storage_dtype: torch.dtype) -> list[dict]:
    converted = []
    for trajectory in group_by_utterance(records):
        converted.extend(build_macro_records(trajectory, storage_dtype))
    return converted


def main() -> int:
    args = parse_args()
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    if source.get("kind") != "cosyvoice3_cfg_velocity_cache":
        raise RuntimeError("Unsupported source trajectory cache")
    if source.get("synthetic_inputs", True):
        raise RuntimeError("Two-step cache requires captured real trajectories")
    storage_dtype = getattr(torch, args.storage_dtype)
    train = convert_split(source["train"], storage_dtype)
    validation = convert_split(source["validation"], storage_dtype)
    midpoint_t = 1.0 - math.cos(math.pi / 4.0)
    payload = {
        "format_version": 1,
        "kind": "cosyvoice3_cfg_velocity_cache",
        "subkind": "two_step_macro_velocity",
        "synthetic_inputs": False,
        "source": str(args.input),
        "solver_steps": 2,
        "time_points": [0.0, midpoint_t, 1.0],
        "storage_dtype": args.storage_dtype,
        "train": train,
        "validation": validation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report = {
        "cache": str(args.output),
        "bytes": args.output.stat().st_size,
        "source": str(args.input),
        "train_records": len(train),
        "validation_records": len(validation),
        "time_points": payload["time_points"],
        "train_target_rms": float(
            torch.sqrt(
                torch.cat([item["target"].float().flatten() for item in train])
                .square()
                .mean()
            )
        ),
        "validation_target_rms": float(
            torch.sqrt(
                torch.cat([item["target"].float().flatten() for item in validation])
                .square()
                .mean()
            )
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
