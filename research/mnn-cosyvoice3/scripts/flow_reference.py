#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort


def stats(values: np.ndarray) -> dict:
    finite = bool(np.isfinite(values).all())
    return {
        "shape": list(values.shape),
        "elements": int(values.size),
        "finite": finite,
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "rms": float(np.sqrt(np.nanmean(np.square(values.astype(np.float64))))),
    }


def create_inputs(sequence_length: int, batch_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260714)
    shape = (batch_size, 80, sequence_length)
    return {
        "x": rng.standard_normal(shape, dtype=np.float32),
        "mask": np.ones((batch_size, 1, sequence_length), dtype=np.float32),
        "mu": rng.standard_normal(shape, dtype=np.float32),
        "t": np.full((batch_size,), 0.5, dtype=np.float32),
        "spks": rng.standard_normal((batch_size, 80), dtype=np.float32),
        "cond": np.zeros(shape, dtype=np.float32),
    }


def run_reference(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = create_inputs(args.sequence_length, args.batch_size)
    for name, values in inputs.items():
        values.tofile(args.output / f"{name}.bin")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    load_start = time.perf_counter()
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    load_ms = (time.perf_counter() - load_start) * 1000.0

    first_start = time.perf_counter()
    output = session.run(["estimator_out"], inputs)[0]
    first_ms = (time.perf_counter() - first_start) * 1000.0
    steady_ms = []
    for _ in range(args.loops):
        started = time.perf_counter()
        output = session.run(["estimator_out"], inputs)[0]
        steady_ms.append((time.perf_counter() - started) * 1000.0)

    output = np.ascontiguousarray(output, dtype=np.float32)
    output.tofile(args.output / "onnx_output.bin")
    metrics = {
        "runtime": "onnxruntime-cpu",
        "onnxruntimeVersion": ort.__version__,
        "sequenceLength": args.sequence_length,
        "batchSize": args.batch_size,
        "threads": args.threads,
        "loadMs": load_ms,
        "firstRunMs": first_ms,
        "steadyAverageMs": float(np.mean(steady_ms)),
        "steadyMedianMs": float(np.median(steady_ms)),
        "output": stats(output),
    }
    (args.output / "onnx_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=True, indent=2), flush=True)
    return 0 if metrics["output"]["finite"] else 2


def compare(args: argparse.Namespace) -> int:
    reference = np.fromfile(args.reference, dtype=np.float32)
    candidate = np.fromfile(args.candidate, dtype=np.float32)
    if reference.size != candidate.size:
        raise RuntimeError(
            f"Element count mismatch: reference={reference.size}, candidate={candidate.size}"
        )
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    denominator = np.maximum(np.abs(reference.astype(np.float64)), 1e-6)
    report = {
        "elements": int(reference.size),
        "referenceFinite": bool(np.isfinite(reference).all()),
        "candidateFinite": bool(np.isfinite(candidate).all()),
        "maxAbs": float(np.max(difference)),
        "meanAbs": float(np.mean(difference)),
        "rmsError": float(np.sqrt(np.mean(np.square(difference)))),
        "maxRelative": float(np.max(difference / denominator)),
        "withinAbsToleranceRatio": float(np.mean(difference <= args.abs_tolerance)),
        "absTolerance": args.abs_tolerance,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    passed = (
        report["referenceFinite"]
        and report["candidateFinite"]
        and report["maxAbs"] <= args.abs_tolerance
    )
    return 0 if passed else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--sequence-length", type=int, default=32)
    run_parser.add_argument("--batch-size", type=int, default=2)
    run_parser.add_argument("--threads", type=int, default=6)
    run_parser.add_argument("--loops", type=int, default=3)
    run_parser.set_defaults(handler=run_reference)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path, required=True)
    compare_parser.add_argument("--abs-tolerance", type=float, default=1e-4)
    compare_parser.set_defaults(handler=compare)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
