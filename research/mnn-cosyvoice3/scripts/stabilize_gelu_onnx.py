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
    clip_min = "cosyvoice_stable_gelu_clip_min"
    clip_max = "cosyvoice_stable_gelu_clip_max"
    existing_names = {initializer.name for initializer in graph.initializer}
    if clip_min not in existing_names:
        graph.initializer.append(helper.make_tensor(clip_min, TensorProto.FLOAT, [], [-10.0]))
    if clip_max not in existing_names:
        graph.initializer.append(helper.make_tensor(clip_max, TensorProto.FLOAT, [], [10.0]))

    nodes = list(graph.node)
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    rewritten = []
    replaced = 0
    for index, node in enumerate(nodes):
        is_gelu_square = (
            node.op_type == "Mul"
            and node.name.endswith("/ff.0.1/Mul")
            and len(node.input) == 2
            and node.input[0] == node.input[1]
            and len(node.output) == 1
        )
        if not is_gelu_square:
            rewritten.append(node)
            continue

        source = node.input[0]
        square_output = node.output[0]
        cube_candidates = [
            candidate
            for candidate in consumers.get(square_output, [])
            if candidate.op_type == "Mul"
            and source in candidate.input
            and candidate.name.endswith("/ff.0.1/Mul_1")
        ]
        if len(cube_candidates) != 1:
            raise RuntimeError(f"Cannot identify GELU cube node after {node.name}")
        cube = cube_candidates[0]
        clipped = f"cosyvoice_stable_gelu_{index}_clipped"
        rewritten.append(
            helper.make_node(
                "Clip",
                [source, clip_min, clip_max],
                [clipped],
                name=f"{node.name}/StableClip",
            )
        )
        node.input[0] = clipped
        node.input[1] = clipped
        for input_index, input_name in enumerate(cube.input):
            if input_name == source:
                cube.input[input_index] = clipped
        rewritten.append(node)
        replaced += 1

    if replaced == 0:
        raise RuntimeError("No tanh-GELU cubic paths found")
    graph.ClearField("node")
    graph.node.extend(rewritten)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"Stabilized {replaced} tanh-GELU cubic paths: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

