#!/usr/bin/env python3
"""Load and optionally run inference with an RT-DETR PyTorch checkpoint.

Examples:
    python3 scripts/load_rtdetr_pt.py --model rtdetr-l.pt
    python3 scripts/load_rtdetr_pt.py --model /path/to/rtdetr-l.pt --source path/to/image.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load an RT-DETR .pt model and optionally run inference.")
    parser.add_argument(
        "--model",
        default="rtdetr-l.pt",
        help=(
            "RT-DETR checkpoint name or path. "
            "Use a local .pt file or a pretrained Ultralytics name such as rtdetr-l.pt."
        ),
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional image, video, or directory path to run a quick inference smoke test.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Inference image size used when --source is provided.",
    )
    return parser.parse_args()


def resolve_model_arg(model_arg: str) -> str:
    model_path = Path(model_arg)
    if model_path.exists():
        return str(model_path.resolve())
    return model_arg


def main() -> None:
    args = parse_args()
    model_arg = resolve_model_arg(args.model)

    print(f"Loading RT-DETR model: {model_arg}")
    model = RTDETR(model_arg)

    names = getattr(model.model, "names", None)
    if names:
        print(f"Loaded classes: {len(names)}")
    else:
        print("Loaded model successfully.")

    if args.source is None:
        print("No --source provided, skipping inference.")
        return

    print(f"Running inference on: {args.source}")
    results = model.predict(source=args.source, imgsz=args.imgsz, verbose=False)

    print(f"Inference completed on {len(results)} item(s).")
    for index, result in enumerate(results[:3]):
        boxes = getattr(result, "boxes", None)
        count = 0 if boxes is None else len(boxes)
        print(f"  result[{index}]: {count} detections")


if __name__ == "__main__":
    main()