#!/usr/bin/env python3
"""Inspect and optionally modify ONNX model to support dynamic input shapes."""

import onnx
from onnx import shape_inference
import argparse

def inspect_model(model_path):
    """Load and inspect ONNX model."""
    model = onnx.load(model_path)

    print(f"Model IR version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print("\nInputs:")
    for inp in model.graph.input:
        print(f"  {inp.name}: {inp.type}")
        if inp.type.HasField('tensor_type'):
            shape = inp.type.tensor_type.shape
            dims = [f"{d.dim_value if d.HasField('dim_value') else d.dim_param}"
                    for d in shape.dim]
            print(f"    Shape: {dims}")

    print("\nOutputs:")
    for out in model.graph.output:
        print(f"  {out.name}: {out.type}")
        if out.type.HasField('tensor_type'):
            shape = out.type.tensor_type.shape
            dims = [f"{d.dim_value if d.HasField('dim_value') else d.dim_param}"
                    for d in shape.dim]
            print(f"    Shape: {dims}")

    return model

def make_dynamic(model_path, output_path, height_dim="height", width_dim="width"):
    """Make input dimensions dynamic."""
    model = onnx.load(model_path)

    # Modify input shape to be dynamic
    for inp in model.graph.input:
        if inp.type.HasField('tensor_type'):
            shape = inp.type.tensor_type.shape
            # Assuming shape is [batch, channels, height, width]
            if len(shape.dim) == 4:
                print(f"Modifying input {inp.name}")
                print(f"  Original shape: {[d.dim_value for d in shape.dim]}")

                # Keep batch=1 and channels=3, make H and W dynamic
                shape.dim[0].dim_value = 1
                shape.dim[1].dim_value = 3
                shape.dim[2].dim_param = height_dim
                shape.dim[3].dim_param = width_dim

                print(f"  New shape: [1, 3, {height_dim}, {width_dim}]")

    # Run shape inference to propagate changes
    print("\nRunning shape inference...")
    try:
        model = shape_inference.infer_shapes(model)
        print("Shape inference successful")
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}")
        print("Saving model anyway...")

    # Save modified model
    onnx.save(model, output_path)
    print(f"\nSaved modified model to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Inspect/modify ONNX model")
    parser.add_argument("model", help="Path to ONNX model")
    parser.add_argument("--make-dynamic", help="Output path for dynamic model")

    args = parser.parse_args()

    print(f"Inspecting {args.model}\n")
    print("=" * 60)

    _model = inspect_model(args.model)

    if args.make_dynamic:
        print("\n" + "=" * 60)
        print("Creating dynamic version...")
        print("=" * 60)
        make_dynamic(args.model, args.make_dynamic)

if __name__ == "__main__":
    main()
