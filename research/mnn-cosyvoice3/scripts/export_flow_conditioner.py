#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from hyperpyyaml import load_hyperpyyaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--utterance-id", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=Path, required=True)
    parser.add_argument("--target-tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def log(stage: str) -> None:
    print(f"[export-flow-conditioner] {stage}", flush=True)


def read_csv(path: Path) -> list[int]:
    return [int(item) for item in path.read_text(encoding="utf-8").replace("\n", "").split(",") if item]


def load_flow(repo: Path, model: Path) -> torch.nn.Module:
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
    return flow.eval().float()


class FlowConditioner(torch.nn.Module):
    def __init__(self, flow: torch.nn.Module):
        super().__init__()
        self.input_embedding = flow.input_embedding
        self.pre_lookahead_layer = flow.pre_lookahead_layer
        self.token_mel_ratio = flow.token_mel_ratio

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.input_embedding(torch.clamp(tokens, min=0))
        encoded = self.pre_lookahead_layer(embedded)
        encoded = encoded.repeat_interleave(self.token_mel_ratio, dim=1)
        return encoded.transpose(1, 2).contiguous()


def find_initial(cache: dict, utterance_id: int) -> dict:
    records = [
        item
        for split in ("train", "validation")
        for item in cache[split]
        if int(item["utterance_id"]) == utterance_id
    ]
    if not records:
        raise RuntimeError(f"Utterance {utterance_id} not found")
    return min(records, key=lambda item: int(item["solver_step"]))


def save_float(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().float().cpu().contiguous().numpy().astype("<f4").tofile(path)


def save_int32(path: Path, values: list[int]) -> None:
    np.asarray(values, dtype="<i4").tofile(path)


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prompt_tokens = read_csv(args.prompt_tokens)
    target_tokens = read_csv(args.target_tokens)
    all_tokens = prompt_tokens + target_tokens
    token_tensor = torch.tensor([all_tokens], dtype=torch.int64)

    log("load_cache_begin")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    initial = find_initial(cache, args.utterance_id)
    prompt_activity = initial["cond"].float().abs().sum(dim=1).squeeze(0)
    nonzero = torch.nonzero(prompt_activity > 0, as_tuple=False).flatten()
    prompt_frames = int(nonzero[-1].item() + 1) if nonzero.numel() else 0
    if prompt_frames != len(prompt_tokens) * 2:
        raise RuntimeError(
            f"Prompt mismatch: {len(prompt_tokens)} tokens but {prompt_frames} mel frames"
        )
    log("load_cache_done")

    log("load_flow_begin")
    flow = load_flow(args.repo, args.model)
    conditioner = FlowConditioner(flow).eval()
    log("load_flow_done")
    mu = conditioner(token_tensor)
    sequence_length = int(mu.shape[2])
    target_frames = sequence_length - prompt_frames

    onnx_path = args.output / "flow-conditioner.fp32.onnx"
    log("export_onnx_begin")
    torch.onnx.export(
        conditioner,
        token_tensor,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["tokens"],
        output_names=["mu"],
        dynamic_axes={"tokens": {1: "token_length"}, "mu": {2: "mel_length"}},
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    candidate = session.run(["mu"], {"tokens": token_tensor.numpy()})[0]
    reference = mu.numpy()
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    log("export_onnx_done")

    prompt_cond = initial["cond"].float()[:, :, :prompt_frames]
    spks = initial["spks"].float()
    rand_noise = flow.decoder.rand_noise.float()
    save_int32(args.output / "prompt-tokens.bin", prompt_tokens)
    save_int32(args.output / "target-tokens.bin", target_tokens)
    save_float(args.output / "prompt-cond.bin", prompt_cond)
    save_float(args.output / "spks.bin", spks)
    save_float(args.output / "rand-noise.bin", rand_noise)
    save_float(args.output / "mu-pytorch.bin", mu)

    report = {
        "format_version": 1,
        "prompt_tokens": len(prompt_tokens),
        "target_tokens": len(target_tokens),
        "total_tokens": len(all_tokens),
        "prompt_frames": prompt_frames,
        "sequence_length": sequence_length,
        "target_frames": target_frames,
        "target_audio_seconds": target_frames * 480 / 24000,
        "rand_noise_shape": list(rand_noise.shape),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_finite": bool(np.isfinite(candidate).all()),
        "onnx_max_abs_error": float(difference.max()),
        "onnx_mean_abs_error": float(difference.mean()),
    }
    (args.output / "conditioner.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["onnx_finite"] or report["onnx_max_abs_error"] > 0.001:
        raise RuntimeError("Flow conditioner ONNX validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
