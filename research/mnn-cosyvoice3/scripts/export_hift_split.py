#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import time

import librosa
import numpy as np
import onnx
import onnxruntime as ort
from scipy.io import wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


SAMPLE_RATE = 24000
N_FFT = 16
HOP_LENGTH = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosyvoice-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference-wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def add_source_paths(source: Path):
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "third_party" / "Matcha-TTS"))


def create_model():
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


def freeze_parametrized_weights(module: nn.Module):
    for child in module.modules():
        if parametrize.is_parametrized(child, "weight"):
            parametrize.remove_parametrizations(child, "weight", leave_parametrized=True)


class F0Network(nn.Module):
    def __init__(self, predictor: nn.Module):
        super().__init__()
        self.predictor = predictor

    def forward(self, mel: torch.Tensor):
        return self.predictor(mel, finalize=True)


class HiFTConvolutionCore(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.conv_pre = model.conv_pre
        self.ups = model.ups
        self.source_downs = model.source_downs
        self.source_resblocks = model.source_resblocks
        self.resblocks = model.resblocks
        self.conv_post = model.conv_post
        self.reflection_pad = model.reflection_pad
        self.num_upsamples = model.num_upsamples
        self.num_kernels = model.num_kernels
        self.lrelu_slope = model.lrelu_slope

    def forward(self, mel: torch.Tensor, source_stft: torch.Tensor):
        x = self.conv_pre(mel)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            source = self.source_downs[i](source_stft)
            source = self.source_resblocks[i](source)
            x = x + source

            combined = self.resblocks[i * self.num_kernels](x)
            for j in range(1, self.num_kernels):
                combined = combined + self.resblocks[i * self.num_kernels + j](x)
            x = combined / self.num_kernels

        x = F.leaky_relu(x)
        return self.conv_post(x)


def read_wave(path: Path):
    sample_rate, data = wavfile.read(path)
    if data.ndim == 2:
        data = data.astype(np.float32).mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        scale = float(max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max))
        data = data.astype(np.float32) / scale
    else:
        data = data.astype(np.float32)
    if sample_rate != SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    peak = float(np.max(np.abs(data)))
    if peak > 1.0:
        data = data / peak
    return torch.from_numpy(data).unsqueeze(0)


def wave_to_log_mel(wave: torch.Tensor):
    n_fft = 1920
    hop = 480
    win = 1920
    padded = F.pad(
        wave.unsqueeze(1),
        ((n_fft - hop) // 2, (n_fft - hop) // 2),
        mode="reflect",
    ).squeeze(1)
    spectrum = torch.stft(
        padded,
        n_fft,
        hop_length=hop,
        win_length=win,
        window=torch.hann_window(win),
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1) + 1e-9)
    mel_filter = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=n_fft,
        n_mels=80,
        fmin=0,
        fmax=None,
    )
    mel = torch.from_numpy(mel_filter).float() @ magnitude
    return torch.log(torch.clamp(mel, min=1e-5))


def create_source_and_stft(model, mel, f0):
    source = model.f0_upsamp(f0[:, None]).transpose(1, 2)
    source, _, _ = model.m_source(source)
    source = source.transpose(1, 2)
    real, imag = model._stft(source.squeeze(1))
    return source, torch.cat([real, imag], dim=1)


def spectral_to_wave(model, spectral):
    split = N_FFT // 2 + 1
    magnitude = torch.clip(torch.exp(spectral[:, :split, :]), max=1e2)
    phase = torch.sin(spectral[:, split:, :])
    return model._istft(magnitude, phase).clamp(-model.audio_limit, model.audio_limit)


def export_onnx(module, inputs, output_path, input_names, output_names, dynamic_axes):
    torch.onnx.export(
        module,
        inputs,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    model = onnx.load(output_path, load_external_data=False)
    onnx.checker.check_model(model)


def compare(reference: np.ndarray, candidate: np.ndarray):
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "rms": float(np.sqrt(np.mean(np.square(delta)))),
        "finite": bool(np.isfinite(candidate).all()),
    }


def save_float(path: Path, tensor: torch.Tensor):
    tensor.detach().cpu().contiguous().numpy().astype(np.float32).tofile(path)


def main():
    args = parse_args()
    add_source_paths(args.cosyvoice_source.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(1986)

    model = create_model()
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().float()
    freeze_parametrized_weights(model)

    wave = read_wave(args.reference_wav)
    mel = wave_to_log_mel(wave).float()
    f0_network = F0Network(model.f0_predictor).eval()
    core = HiFTConvolutionCore(model).eval()

    started = time.perf_counter()
    with torch.inference_mode():
        f0 = f0_network(mel)
        source, source_stft = create_source_and_stft(model, mel, f0)
        spectral = core(mel, source_stft)
        reconstructed = spectral_to_wave(model, spectral)
    pytorch_seconds = time.perf_counter() - started

    f0_path = args.output / "hift-f0.fp32.onnx"
    core_path = args.output / "hift-core.fp32.onnx"
    export_onnx(
        f0_network,
        mel,
        f0_path,
        ["mel"],
        ["f0"],
        {"mel": {2: "mel_frames"}, "f0": {1: "mel_frames"}},
    )
    export_onnx(
        core,
        (mel, source_stft),
        core_path,
        ["mel", "source_stft"],
        ["spectral"],
        {
            "mel": {2: "mel_frames"},
            "source_stft": {2: "source_frames"},
            "spectral": {2: "source_frames"},
        },
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    f0_session = ort.InferenceSession(str(f0_path), options, providers=["CPUExecutionProvider"])
    core_session = ort.InferenceSession(str(core_path), options, providers=["CPUExecutionProvider"])
    mel_np = mel.numpy().astype(np.float32)
    source_stft_np = source_stft.numpy().astype(np.float32)
    f0_ort = f0_session.run(None, {"mel": mel_np})[0]
    spectral_ort = core_session.run(
        None,
        {"mel": mel_np, "source_stft": source_stft_np},
    )[0]
    reconstructed_ort = spectral_to_wave(model, torch.from_numpy(spectral_ort))

    save_float(args.output / "mel.bin", mel)
    save_float(args.output / "source-stft.bin", source_stft)
    save_float(args.output / "f0-pytorch.bin", f0)
    save_float(args.output / "spectral-pytorch.bin", spectral)
    save_float(args.output / "pcm-pytorch.bin", reconstructed)
    save_float(args.output / "pcm-onnx.bin", reconstructed_ort)
    wavfile.write(
        args.output / "reference-resampled.wav",
        SAMPLE_RATE,
        np.clip(wave.squeeze(0).numpy(), -1.0, 1.0),
    )
    wavfile.write(
        args.output / "hift-pytorch.wav",
        SAMPLE_RATE,
        reconstructed.squeeze(0).numpy().astype(np.float32),
    )
    wavfile.write(
        args.output / "hift-onnx.wav",
        SAMPLE_RATE,
        reconstructed_ort.squeeze(0).numpy().astype(np.float32),
    )

    report = {
        "sample_rate": SAMPLE_RATE,
        "source_samples": int(wave.shape[-1]),
        "mel_shape": list(mel.shape),
        "source_stft_shape": list(source_stft.shape),
        "spectral_shape": list(spectral.shape),
        "pcm_samples": int(reconstructed.shape[-1]),
        "audio_seconds": reconstructed.shape[-1] / SAMPLE_RATE,
        "pytorch_seconds": pytorch_seconds,
        "pytorch_rtf": pytorch_seconds / (reconstructed.shape[-1] / SAMPLE_RATE),
        "f0_onnx_error": compare(f0.numpy(), f0_ort),
        "core_onnx_error": compare(spectral.numpy(), spectral_ort),
        "pcm_onnx_error": compare(reconstructed.numpy(), reconstructed_ort.numpy()),
        "pcm_peak": float(reconstructed.abs().max()),
        "pcm_rms": float(reconstructed.square().mean().sqrt()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["core_onnx_error"]["finite"] or report["core_onnx_error"]["max_abs"] > 0.02:
        raise RuntimeError("HiFT ONNX numerical gate failed")


if __name__ == "__main__":
    main()
