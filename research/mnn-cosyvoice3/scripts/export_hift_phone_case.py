#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import torch
from torch.nn.utils import parametrize


SAMPLE_RATE = 24000
N_FFT = 16
HOP_LENGTH = 4
UPSAMPLE_SCALE = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosyvoice-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mel-bin", type=Path, required=True)
    parser.add_argument("--mel-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def log(stage: str) -> None:
    print(f"[export-hift-phone-case] {stage}", flush=True)


def add_source_paths(source: Path) -> None:
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "third_party" / "Matcha-TTS"))


def create_model() -> torch.nn.Module:
    from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
    from cosyvoice.hifigan.generator import CausalHiFTGenerator

    return CausalHiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=SAMPLE_RATE,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": N_FFT, "hop_len": HOP_LENGTH},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5]] * 3,
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5]] * 3,
        lrelu_slope=0.1,
        audio_limit=0.99,
        conv_pre_look_right=4,
        f0_predictor=CausalConvRNNF0Predictor(
            num_class=1,
            in_channels=80,
            cond_channels=512,
        ),
    )


def freeze_parametrized_weights(module: torch.nn.Module) -> None:
    for child in module.modules():
        if parametrize.is_parametrized(child, "weight"):
            parametrize.remove_parametrizations(
                child, "weight", leave_parametrized=True
            )


def save_float(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().cpu().contiguous().numpy().astype("<f4").tofile(path)


def stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all()),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "rms": float(torch.sqrt(value.square().mean())),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    add_source_paths(args.cosyvoice_source.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(1986)

    log("load_model_begin")
    model = create_model()
    model.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True), strict=True
    )
    model.eval().float()
    freeze_parametrized_weights(model)
    log("load_model_done")

    mel_np = np.fromfile(args.mel_bin, dtype="<f4")
    expected = 80 * args.mel_frames
    if mel_np.size != expected:
        raise RuntimeError(f"Expected {expected} mel values, got {mel_np.size}")
    mel = torch.from_numpy(mel_np.reshape(1, 80, args.mel_frames).copy())

    log("inference_begin")
    started = time.perf_counter()
    f0 = model.f0_predictor(mel, finalize=True)
    f0_upsampled = model.f0_upsamp(f0[:, None]).transpose(1, 2)
    sine_waves, uv, sine_noise = model.m_source.l_sin_gen(f0_upsampled)
    source = model.m_source.l_tanh(model.m_source.l_linear(sine_waves))
    source = source.transpose(1, 2)
    source_real, source_imag = model._stft(source.squeeze(1))
    source_stft = torch.cat([source_real, source_imag], dim=1)

    x = model.conv_pre(mel)
    for i in range(model.num_upsamples):
        x = torch.nn.functional.leaky_relu(x, model.lrelu_slope)
        x = model.ups[i](x)
        if i == model.num_upsamples - 1:
            x = model.reflection_pad(x)
        source_branch = model.source_downs[i](source_stft)
        source_branch = model.source_resblocks[i](source_branch)
        x = x + source_branch
        combined = model.resblocks[i * model.num_kernels](x)
        for j in range(1, model.num_kernels):
            combined = combined + model.resblocks[i * model.num_kernels + j](x)
        x = combined / model.num_kernels
    spectral = model.conv_post(torch.nn.functional.leaky_relu(x))
    split = N_FFT // 2 + 1
    magnitude = torch.clip(torch.exp(spectral[:, :split]), max=100.0)
    phase = torch.sin(spectral[:, split:])
    pcm = model._istft(magnitude, phase).clamp(-model.audio_limit, model.audio_limit)
    inference_seconds = time.perf_counter() - started
    log("inference_done")

    sine_generator = model.m_source.l_sin_gen
    source_samples = args.mel_frames * UPSAMPLE_SCALE
    save_float(args.output / "mel.bin", mel)
    save_float(args.output / "f0.bin", f0)
    save_float(args.output / "f0-upsampled.bin", f0_upsampled)
    save_float(args.output / "sine-waves.bin", sine_waves)
    save_float(args.output / "sine-noise.bin", sine_noise)
    save_float(args.output / "source.bin", source)
    save_float(args.output / "source-stft.bin", source_stft)
    save_float(args.output / "spectral.bin", spectral)
    save_float(args.output / "pcm.bin", pcm)
    save_float(args.output / "source-rand-ini.bin", sine_generator.rand_ini)
    save_float(
        args.output / "source-noise-table.bin",
        sine_generator.sine_waves[:, :source_samples],
    )
    save_float(args.output / "source-linear-weight.bin", model.m_source.l_linear.weight)
    save_float(args.output / "source-linear-bias.bin", model.m_source.l_linear.bias)
    save_float(args.output / "uv.bin", uv)
    wavfile.write(
        args.output / "hift-reference.wav",
        SAMPLE_RATE,
        pcm.squeeze(0).numpy().astype(np.float32),
    )

    report = {
        "format_version": 1,
        "sample_rate": SAMPLE_RATE,
        "mel_frames": args.mel_frames,
        "source_samples": source_samples,
        "source_frames": args.mel_frames * 120 + 1,
        "pcm_samples": int(pcm.shape[-1]),
        "audio_seconds": float(pcm.shape[-1] / SAMPLE_RATE),
        "inference_seconds": inference_seconds,
        "rtf": inference_seconds / (pcm.shape[-1] / SAMPLE_RATE),
        "tensors": {
            "mel": stats(mel),
            "f0": stats(f0),
            "f0_upsampled": stats(f0_upsampled),
            "sine_waves": stats(sine_waves),
            "source": stats(source),
            "source_stft": stats(source_stft),
            "spectral": stats(spectral),
            "pcm": stats(pcm),
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
