#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral", type=Path, required=True)
    parser.add_argument("--mel-frames", type=int, required=True)
    parser.add_argument("--reference-pcm", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--abs-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    source_frames = args.mel_frames * 120 + 1
    spectral_np = np.fromfile(args.spectral, dtype=np.float32)
    expected = 18 * source_frames
    if spectral_np.size != expected:
        raise RuntimeError(f"spectral elements: expected {expected}, got {spectral_np.size}")
    spectral = torch.from_numpy(spectral_np.reshape(1, 18, source_frames))
    magnitude = torch.clip(torch.exp(spectral[:, :9]), max=1e2)
    phase = torch.sin(spectral[:, 9:])
    complex_spectrum = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
    pcm = torch.istft(
        complex_spectrum,
        n_fft=16,
        hop_length=4,
        win_length=16,
        window=torch.hann_window(16),
    ).clamp(-0.99, 0.99).numpy().astype(np.float32)
    reference = np.fromfile(args.reference_pcm, dtype=np.float32).reshape(1, -1)
    if pcm.shape != reference.shape:
        raise RuntimeError(f"PCM shape mismatch: {pcm.shape} != {reference.shape}")
    delta = np.abs(pcm.astype(np.float64) - reference.astype(np.float64))
    report = {
        "finite": bool(np.isfinite(pcm).all()),
        "samples": int(pcm.size),
        "audioSeconds": pcm.size / 24000.0,
        "peak": float(np.max(np.abs(pcm))),
        "rms": float(np.sqrt(np.mean(np.square(pcm.astype(np.float64))))),
        "dc": float(np.mean(pcm)),
        "clippedRatio": float(np.mean(np.abs(pcm) >= 0.989)),
        "maxAbsError": float(delta.max()),
        "meanAbsError": float(delta.mean()),
        "rmsError": float(np.sqrt(np.mean(np.square(delta)))),
        "absTolerance": args.abs_tolerance,
    }
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(args.output_wav, 24000, pcm.squeeze(0))
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["finite"] or report["maxAbsError"] > args.abs_tolerance:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
