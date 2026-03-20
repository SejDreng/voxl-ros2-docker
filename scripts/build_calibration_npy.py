import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def find_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    if recursive:
        for pattern in patterns:
            yield from input_dir.rglob(pattern)
            yield from input_dir.rglob(pattern.upper())
    else:
        for pattern in patterns:
            yield from input_dir.glob(pattern)
            yield from input_dir.glob(pattern.upper())


def letterbox_rgb(image_rgb: np.ndarray, target_size: int = 640, pad_value: int = 114) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = np.array(
        Image.fromarray(image_rgb).resize((new_w, new_h), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )

    canvas = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def preprocess_image(image_path: Path, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    rgb = np.array(image, dtype=np.uint8)
    letterboxed = letterbox_rgb(rgb, target_size=image_size)

    nhwc = letterboxed.astype(np.float32) / 255.0
    return nhwc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build calibration_data.npy for ONNX->TFLite INT8 quantization (YOLOv8 style)."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing calibration images.")
    parser.add_argument(
        "--output",
        default="quantization_data/calibration_data.npy",
        help="Output .npy file path.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to include. If omitted, all discovered images are used.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Square image size expected by model (default: 640).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for images.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = sorted(set(find_images(input_dir, recursive=args.recursive)))
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    selected_paths = image_paths if args.max_images is None else image_paths[: args.max_images]

    batch = []
    for image_path in selected_paths:
        try:
            batch.append(preprocess_image(image_path, image_size=args.image_size))
        except Exception as exc:
            print(f"Skipping {image_path}: {exc}")

    if not batch:
        raise RuntimeError("No valid images were processed.")

    calibration_array = np.stack(batch, axis=0).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, calibration_array)

    print(f"Saved: {output_path}")
    print(f"Shape: {calibration_array.shape} (N, H, W, C)")
    print(f"Dtype: {calibration_array.dtype}")
    print(f"Range: [{calibration_array.min():.4f}, {calibration_array.max():.4f}]")
    print(f"Images used: {len(batch)}")


if __name__ == "__main__":
    main()
