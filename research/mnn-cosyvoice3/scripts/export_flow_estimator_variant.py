import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from hyperpyyaml import load_hyperpyyaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dict-patch", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    return parser.parse_args()


def make_inputs(batch_size, sequence_length, channels):
    generator = torch.Generator().manual_seed(20260715)
    shape = (batch_size, channels, sequence_length)
    return (
        torch.randn(shape, generator=generator),
        torch.ones((batch_size, 1, sequence_length)),
        torch.randn(shape, generator=generator),
        torch.full((batch_size,), 0.5),
        torch.randn((batch_size, channels), generator=generator),
        torch.zeros(shape),
    )


def main():
    args = parse_args()
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "third_party" / "Matcha-TTS"))

    with (args.model / "cosyvoice3.yaml").open("r", encoding="utf-8") as stream:
        configs = load_hyperpyyaml(
            stream,
            overrides={"qwen_pretrain_path": str(args.model / "CosyVoice-BlankEN")},
        )
    flow = configs["flow"]
    flow.load_state_dict(
        torch.load(args.model / "flow.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    estimator = flow.decoder.estimator.eval().float()
    patch_info = None
    if args.state_dict_patch is not None:
        payload = torch.load(args.state_dict_patch, map_location="cpu", weights_only=False)
        if payload.get("kind") != "cosyvoice3_cfg_student_patch":
            raise RuntimeError("Unsupported estimator state-dict patch")
        state_dict_patch = {
            name: value.float() for name, value in payload["state_dict_patch"].items()
        }
        incompatible = estimator.load_state_dict(state_dict_patch, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected estimator patch keys: {incompatible.unexpected_keys}"
            )
        patch_info = {
            "path": str(args.state_dict_patch),
            "parameter_tensors": len(state_dict_patch),
            "last_blocks": payload.get("last_blocks"),
        }
    channels = estimator.out_channels
    inputs = make_inputs(args.batch_size, args.sequence_length, channels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        reference = estimator(*inputs).detach().cpu().numpy()
    torch.onnx.export(
        estimator,
        inputs,
        args.output,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["x", "mask", "mu", "t", "spks", "cond"],
        output_names=["estimator_out"],
        dynamic_axes={
            "x": {2: "seq_len"},
            "mask": {2: "seq_len"},
            "mu": {2: "seq_len"},
            "cond": {2: "seq_len"},
            "estimator_out": {2: "seq_len"},
        },
    )

    session = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    candidate = session.run(
        ["estimator_out"],
        {name: value.numpy() for name, value in zip(
            ["x", "mask", "mu", "t", "spks", "cond"], inputs
        )},
    )[0]
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    report = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "state_dict_patch": patch_info,
        "onnx": str(args.output),
        "onnx_bytes": args.output.stat().st_size,
        "finite": bool(np.isfinite(candidate).all()),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "rms_error": float(np.sqrt(np.mean(np.square(difference)))),
    }
    report_path = args.output.with_suffix(".validation.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["finite"] or report["max_abs_error"] > 0.02:
        raise RuntimeError("Exported Flow estimator failed numerical validation")


if __name__ == "__main__":
    main()
