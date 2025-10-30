#!/usr/bin/env python
"""
Generate OpenCV YuNet parity fixtures for the images under fixtures/images/.

Example:
    python generate_opencv_parity.py \
        --images ../fixtures/images \
        --output ../fixtures/opencv \
        --model ../models/face_detection_yunet_2023mar.onnx \
        --pattern "*"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


DEFAULT_INPUT_SIZE = (320, 320)
DEFAULT_SCORE_THRESHOLD = 0.9
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_TOP_K = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OpenCV YuNet parity fixtures.")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("../fixtures/images"),
        help="Directory that contains input images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../fixtures/opencv"),
        help="Directory to write JSON fixtures.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("../models/face_detection_yunet_2023mar.onnx"),
        help="Path to YuNet ONNX model.",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern relative to --images to select images (default: '*').",
    )
    parser.add_argument(
        "--input-size",
        default="320x320",
        help="Inference input size as WIDTHxHEIGHT (default: 320x320).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Score threshold passed to YuNet (default: 0.9).",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=DEFAULT_NMS_THRESHOLD,
        help="NMS threshold passed to YuNet (default: 0.3).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-K parameter passed to YuNet (default: 5000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the images that would be processed without writing JSON.",
    )
    return parser.parse_args()


def parse_size(value: str) -> Tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid size '{value}', expected WIDTHxHEIGHT.")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer in size '{value}'.") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Input dimensions must be positive.")
    return width, height


def collect_images(root: Path, pattern: str) -> List[Path]:
    return sorted(p for p in root.glob(pattern) if p.is_file())


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_detector(
    model_path: Path,
    input_size: Tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
    top_k: int,
) -> cv2.FaceDetectorYN:
    detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        input_size,
        score_threshold,
        nms_threshold,
        top_k,
    )
    return detector


def run_detector(detector: cv2.FaceDetectorYN, image: np.ndarray) -> np.ndarray:
    _, detections = detector.detect(image)
    if detections is None:
        return np.empty((0, 15), dtype=np.float32)
    return detections


def detection_rows_to_json(detections: np.ndarray) -> List[dict]:
    rows: List[dict] = []
    for row in detections:
        # Each row: [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y, rm_x, rm_y, lm_x, lm_y, score]
        bbox = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
        landmarks = [
            [float(row[4]), float(row[5])],
            [float(row[6]), float(row[7])],
            [float(row[8]), float(row[9])],
            [float(row[10]), float(row[11])],
            [float(row[12]), float(row[13])],
        ]
        score = float(row[14])
        rows.append(
            {
                "score": score,
                "bbox": bbox,
                "landmarks": landmarks,
            }
        )
    return rows


def write_fixture(
    output_path: Path,
    image_path: Path,
    detections: np.ndarray,
    input_size: Tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
    top_k: int,
) -> None:
    data = {
        "image": str(image_path),
        "input_size": list(input_size),
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
        "top_k": top_k,
        "detections": detection_rows_to_json(detections),
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def process_images(args: argparse.Namespace) -> None:
    input_size = parse_size(args.input_size)
    images = collect_images(args.images, args.pattern)
    ensure_output_dir(args.output)

    if args.dry_run:
        for image_path in images:
            print(f"[DRY-RUN] Would process {image_path}")
        return

    if not images:
        print(f"No images found under {args.images} matching '{args.pattern}'.")
        return

    detector = create_detector(
        args.model,
        input_size,
        args.score_threshold,
        args.nms_threshold,
        args.top_k,
    )

    input_w, input_h = input_size

    for image_path in images:
        rel_name = image_path.name
        print(f"Processing {rel_name} ...")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  [WARN] Failed to read image {image_path}, skipping.")
            continue
        orig_h, orig_w = image.shape[:2]
        resized = cv2.resize(image, input_size)
        detector.setInputSize(input_size)
        detections = run_detector(detector, resized)

        if detections.size > 0:
            scale_x = orig_w / float(input_w)
            scale_y = orig_h / float(input_h)
            # Scale bbox coordinates
            detections[:, 0] *= scale_x
            detections[:, 1] *= scale_y
            detections[:, 2] *= scale_x
            detections[:, 3] *= scale_y
            # Scale landmarks
            detections[:, 4] *= scale_x
            detections[:, 5] *= scale_y
            detections[:, 6] *= scale_x
            detections[:, 7] *= scale_y
            detections[:, 8] *= scale_x
            detections[:, 9] *= scale_y
            detections[:, 10] *= scale_x
            detections[:, 11] *= scale_y
            detections[:, 12] *= scale_x
            detections[:, 13] *= scale_y

        output_path = args.output / f"{Path(rel_name).stem}.json"
        write_fixture(
            output_path,
            image_path,
            detections,
            input_size,
            args.score_threshold,
            args.nms_threshold,
            args.top_k,
        )
        print(f"  Wrote {len(detections)} detection(s) to {output_path}")


def main() -> None:
    args = parse_args()
    process_images(args)


if __name__ == "__main__":
    main()
