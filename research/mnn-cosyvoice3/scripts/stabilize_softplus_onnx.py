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
    existing_names = {initializer.name for initializer in graph.initializer}
    zero_name = "cosyvoice_stable_softplus_zero"
    one_name = "cosyvoice_stable_softplus_one"
    if zero_name not in existing_names:
        graph.initializer.append(helper.make_tensor(zero_name, TensorProto.FLOAT, [], [0.0]))
    if one_name not in existing_names:
        graph.initializer.append(helper.make_tensor(one_name, TensorProto.FLOAT, [], [1.0]))

    rewritten = []
    replaced = 0
    for index, node in enumerate(graph.node):
        if node.op_type != "Softplus":
            rewritten.append(node)
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            raise RuntimeError(f"Unexpected Softplus signature: {node.name}")
        source = node.input[0]
        target = node.output[0]
        prefix = f"cosyvoice_stable_softplus_{index}"
        abs_value = f"{prefix}_abs"
        negative_abs = f"{prefix}_negative_abs"
        exponent = f"{prefix}_exp"
        exponent_plus_one = f"{prefix}_exp_plus_one"
        logarithm = f"{prefix}_log"
        positive = f"{prefix}_positive"
        base_name = node.name or prefix
        rewritten.extend(
            [
                helper.make_node("Abs", [source], [abs_value], name=f"{base_name}/StableAbs"),
                helper.make_node("Neg", [abs_value], [negative_abs], name=f"{base_name}/StableNeg"),
                helper.make_node("Exp", [negative_abs], [exponent], name=f"{base_name}/StableExp"),
                helper.make_node(
                    "Add", [exponent, one_name], [exponent_plus_one], name=f"{base_name}/StableAddOne"
                ),
                helper.make_node("Log", [exponent_plus_one], [logarithm], name=f"{base_name}/StableLog"),
                helper.make_node("Max", [source, zero_name], [positive], name=f"{base_name}/StablePositive"),
                helper.make_node("Add", [positive, logarithm], [target], name=f"{base_name}/StableResult"),
            ]
        )
        replaced += 1

    if replaced == 0:
        raise RuntimeError("No Softplus nodes found")
    graph.ClearField("node")
    graph.node.extend(rewritten)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"Replaced {replaced} Softplus nodes: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

