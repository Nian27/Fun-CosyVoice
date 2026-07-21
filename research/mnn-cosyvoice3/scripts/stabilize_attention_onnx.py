#!/usr/bin/env python3
import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(args.input, load_external_data=True)
    graph = model.graph
    constants = {
        "cosyvoice_attention_scale_down": 1.0 / 64.0,
        "cosyvoice_attention_clip_min": -900.0,
        "cosyvoice_attention_clip_max": 900.0,
        "cosyvoice_attention_scale_up": 64.0,
    }
    existing_names = {initializer.name for initializer in graph.initializer}
    for name, value in constants.items():
        if name not in existing_names:
            graph.initializer.append(helper.make_tensor(name, TensorProto.FLOAT, [], [value]))

    rewritten = []
    replaced = 0
    for index, node in enumerate(graph.node):
        is_attention_scores = (
            node.op_type == "MatMul"
            and node.name.endswith("/attn/MatMul")
            and len(node.input) == 2
            and len(node.output) == 1
        )
        if not is_attention_scores:
            rewritten.append(node)
            continue
        query, key = node.input
        target = node.output[0]
        prefix = f"cosyvoice_stable_attention_{index}"
        scaled_query = f"{prefix}_scaled_query"
        scaled_scores = f"{prefix}_scaled_scores"
        clipped_scores = f"{prefix}_clipped_scores"
        base_name = node.name
        rewritten.extend(
            [
                helper.make_node(
                    "Mul",
                    [query, "cosyvoice_attention_scale_down"],
                    [scaled_query],
                    name=f"{base_name}/ScaleQuery",
                ),
                helper.make_node(
                    "MatMul", [scaled_query, key], [scaled_scores], name=f"{base_name}/ScaledMatMul"
                ),
                helper.make_node(
                    "Clip",
                    [
                        scaled_scores,
                        "cosyvoice_attention_clip_min",
                        "cosyvoice_attention_clip_max",
                    ],
                    [clipped_scores],
                    name=f"{base_name}/ClipScaledScores",
                ),
                helper.make_node(
                    "Mul",
                    [clipped_scores, "cosyvoice_attention_scale_up"],
                    [target],
                    name=f"{base_name}/RestoreScale",
                ),
            ]
        )
        replaced += 1

    if replaced == 0:
        raise RuntimeError("No attention score MatMul nodes found")
    graph.ClearField("node")
    graph.node.extend(rewritten)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"Stabilized {replaced} attention score MatMul nodes: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

