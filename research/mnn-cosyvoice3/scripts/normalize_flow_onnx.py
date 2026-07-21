#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(str(args.input), load_external_data=True)
    existing_names = {initializer.name for initializer in model.graph.initializer}
    changed = 0
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Pad" or len(node.input) < 3 or node.input[2]:
            continue
        name = f"__mnn_explicit_pad_value_{index}"
        if name in existing_names:
            raise RuntimeError(f"Initializer already exists: {name}")
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), name=name)
        )
        node.input[2] = name
        existing_names.add(name)
        changed += 1

    if changed == 0:
        raise RuntimeError("No Pad node with an omitted constant_value input was found")
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(args.output))
    print(f"Normalized {changed} Pad nodes: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
