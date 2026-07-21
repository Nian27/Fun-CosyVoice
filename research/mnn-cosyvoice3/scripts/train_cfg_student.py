#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml


INPUT_NAMES = ("x", "mask", "mu", "t", "spks", "cond")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-patch", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--last-blocks", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
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
    return flow.decoder.estimator


def configure_trainable(estimator: torch.nn.Module, last_blocks: int) -> list[str]:
    if last_blocks < 1 or last_blocks > len(estimator.transformer_blocks):
        raise ValueError("last-blocks is outside the estimator depth")
    for parameter in estimator.parameters():
        parameter.requires_grad = False
    trainable_prefixes = (
        f"transformer_blocks.{len(estimator.transformer_blocks) - last_blocks}",
        "norm_out",
        "proj_out",
    )
    first_trainable = len(estimator.transformer_blocks) - last_blocks
    for index in range(first_trainable, len(estimator.transformer_blocks)):
        for parameter in estimator.transformer_blocks[index].parameters():
            parameter.requires_grad = True
    for module in (estimator.norm_out, estimator.proj_out):
        for parameter in module.parameters():
            parameter.requires_grad = True
    return list(trainable_prefixes)


def move_record(record: dict, device: torch.device) -> tuple[dict, torch.Tensor]:
    inputs = {
        name: record[name].to(device=device, dtype=torch.float32)
        for name in INPUT_NAMES
    }
    target = record["target"].to(device=device, dtype=torch.float32)
    return inputs, target


def masked_mse(candidate: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.expand_as(candidate)
    squared = F.mse_loss(candidate, target, reduction="none") * expanded_mask
    return squared.sum() / expanded_mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(estimator: torch.nn.Module, records: list[dict], device: torch.device) -> float:
    estimator.eval()
    losses = []
    for record in records:
        inputs, target = move_record(record, device)
        candidate = estimator(**inputs)
        losses.append(masked_mse(candidate, target, inputs["mask"]).item())
    return float(np.mean(losses))


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(args.seed)
    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("kind") != "cosyvoice3_cfg_velocity_cache":
        raise RuntimeError("Unsupported teacher cache")

    estimator = load_estimator(args.repo, args.model).to(device=device, dtype=torch.float32)
    initial_patch_info = None
    if args.initial_patch is not None:
        initial_payload = torch.load(
            args.initial_patch, map_location="cpu", weights_only=False
        )
        if initial_payload.get("kind") != "cosyvoice3_cfg_student_patch":
            raise RuntimeError("Unsupported initial student patch")
        state_dict_patch = {
            name: value.float()
            for name, value in initial_payload["state_dict_patch"].items()
        }
        incompatible = estimator.load_state_dict(state_dict_patch, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected initial patch keys: {incompatible.unexpected_keys}"
            )
        initial_patch_info = {
            "path": str(args.initial_patch),
            "parameter_tensors": len(state_dict_patch),
        }
    trainable_prefixes = configure_trainable(estimator, args.last_blocks)
    estimator.eval()
    trainable = [parameter for parameter in estimator.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in estimator.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_train_loss = evaluate(estimator, cache["train"], device)
    initial_validation_loss = evaluate(estimator, cache["validation"], device)
    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        estimator.eval()
        record = cache["train"][(step - 1) % len(cache["train"])]
        inputs, target = move_record(record, device)
        optimizer.zero_grad(set_to_none=True)
        candidate = estimator(**inputs)
        loss = masked_mse(candidate, target, inputs["mask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            validation_loss = evaluate(estimator, cache["validation"], device)
            event = {
                "step": step,
                "train_sample_loss": float(loss.detach().cpu()),
                "validation_loss": validation_loss,
            }
            history.append(event)
            print(json.dumps(event), flush=True)

    elapsed = time.perf_counter() - started
    final_train_loss = evaluate(estimator, cache["train"], device)
    final_validation_loss = evaluate(estimator, cache["validation"], device)
    patch = {
        name: parameter.detach().cpu().to(torch.float16)
        for name, parameter in estimator.named_parameters()
        if parameter.requires_grad
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "kind": "cosyvoice3_cfg_student_patch",
            "base_model": str(args.model),
            "cache": str(args.cache),
            "initial_patch": initial_patch_info,
            "last_blocks": args.last_blocks,
            "state_dict_patch": patch,
        },
        args.output,
    )
    report = {
        "checkpoint": str(args.output),
        "checkpoint_bytes": args.output.stat().st_size,
        "synthetic_inputs": bool(cache.get("synthetic_inputs", False)),
        "cache_subkind": cache.get("subkind"),
        "initial_patch": initial_patch_info,
        "steps": args.steps,
        "last_blocks": args.last_blocks,
        "trainable_prefixes": trainable_prefixes,
        "total_parameters": total_count,
        "trainable_parameters": trainable_count,
        "trainable_ratio": trainable_count / total_count,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / args.steps,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "history": history,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
