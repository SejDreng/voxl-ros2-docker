import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
import torchvision
import torch.onnx
from ultralytics import YOLO
from torchinfo import summary

import os
import argparse
import subprocess
import sys
import shutil
from pathlib import Path
import numpy as np

import tensorflow as tf


if torch.cuda.is_available():
    device = torch.device('cuda')
else:    
    device = torch.device('cpu')

# ====================================Path Configuration========================================== #


# REPLACE WITH YOUR OWN MODEL NAME
MODEL_NAME = 'yolov8n.pt'
MODEL_PREFIX = Path(MODEL_NAME).stem
DUMMY_INPUT_SHAPE = (1, 3, 640, 640)  # Adjust as needed for your model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / 'models'

# REPLACE WITH YOUR OWN CALIBRATION DATA PATH
CALIBRATION_DATA_PATH = MODELS_PATH / 'quantization_data' / 'Images'
CALIBRATION_NPY_PATH = MODELS_PATH / 'quantization_data' / 'calibration_data.npy'

TFLITE_MODELS_PATH = MODELS_PATH / 'saved_models'


# ==================================Specify Model============================================ #

class NNmodel(nn.Module):
    def __init__(self):
        super(NNmodel, self).__init__()
        
        # yolo = YOLO('yolov8n.pt')  # downloads weights if not cached
        # self.model = yolo.model.float().to(device)
        
        # alternative, load yolo model from local file
        model_path = str(MODELS_PATH / MODEL_NAME)
        model = torch.load(model_path, map_location=device, weights_only=False)
        self.model = model['model'].float().to(device)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
# ==================================Export Pipeline============================================ #
    
def get_calibration_npy() -> str | None:
    expected_h = int(DUMMY_INPUT_SHAPE[2])
    expected_w = int(DUMMY_INPUT_SHAPE[3])

    if CALIBRATION_NPY_PATH.is_file():
        try:
            calibration_array = np.load(CALIBRATION_NPY_PATH)
            if (
                calibration_array.ndim == 4
                and calibration_array.shape[-1] == 3
                and calibration_array.shape[1] == expected_h
                and calibration_array.shape[2] == expected_w
            ):
                return str(CALIBRATION_NPY_PATH)

            print(
                f"Existing calibration data has unexpected shape {calibration_array.shape}. "
                f"Regenerating as NHWC {expected_h}x{expected_w} for onnx2tf."
            )
            CALIBRATION_NPY_PATH.unlink(missing_ok=True)
        except Exception as exc:
            print(f"Failed to validate existing calibration npy: {exc}. Regenerating.")
            CALIBRATION_NPY_PATH.unlink(missing_ok=True)

    image_patterns = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    calibration_dir = CALIBRATION_DATA_PATH
    if calibration_dir.is_dir() and any(calibration_dir.rglob(p) for p in image_patterns):
        calibration_script = os.path.join(os.path.dirname(__file__), 'build_calibration_npy.py')
        subprocess.run(
            [sys.executable, calibration_script,
             '--input-dir', str(CALIBRATION_DATA_PATH),
             '--output', str(CALIBRATION_NPY_PATH),
             '--image-size', str(expected_h),
             '--recursive'],
            check=True,
        )
        return str(CALIBRATION_NPY_PATH)
    
    print(f"No calibration images found under: {CALIBRATION_DATA_PATH}")
    return None


def build_onnx2tf_cmd(
    onnx_path: str,
    output_dir: str,
    calibration_npy: str | None,
    *,
    include_quantization: bool,
    include_validation: bool,
) -> list[str]:
    # Always run onnx2tf via the current Python interpreter to avoid
    # broken console-script shebangs in relocated virtual environments.
    cmd = [sys.executable, '-m', 'onnx2tf', '-i', onnx_path, '-o', output_dir, '-b', '1']

    if include_validation:
        cmd += ['-cotof', '-dms']

    if include_quantization and calibration_npy:
        print(f"Using INT8 calibration data: {calibration_npy}")
        calibration_mean = '[[[[0.0,0.0,0.0]]]]'
        calibration_std = '[[[[1.0,1.0,1.0]]]]'
        cmd += [
            '-oiqt',
            '-cind', 'x', calibration_npy, calibration_mean, calibration_std,
            '-iqd', 'int8', '-oqd', 'int8',
        ]
    elif include_quantization:
        print("No calibration data found. Running non-INT8 onnx2tf conversion.")

    if include_quantization and include_validation:
        cmd.append('-agje')

    return cmd


def _auto_json_candidates(onnx_path: str, output_dir: str) -> list[Path]:
    onnx_file = Path(onnx_path)
    out_dir = Path(output_dir)
    parent = onnx_file.parent
    stem = onnx_file.stem
    return [
        out_dir / f'{stem}_auto.json',
        parent / f'{stem}_auto.json',
        Path.cwd() / f'{stem}_auto.json',
    ]


def ensure_onnx2tf_sample_data_file() -> None:
    """Ensure onnx2tf can load local sample data without network/download issues."""
    sample_name = 'calibration_image_sample_data_20x128x128x3_float32.npy'
    sample_path = Path.cwd() / sample_name

    if sample_path.is_file():
        try:
            arr = np.load(sample_path)
            if arr.shape == (20, 128, 128, 3) and arr.dtype == np.float32:
                return
            print(f"Replacing invalid onnx2tf sample data file: {sample_path} (shape={arr.shape}, dtype={arr.dtype})")
        except Exception as exc:
            print(f"Replacing unreadable onnx2tf sample data file: {sample_path} ({exc})")

    # Match onnx2tf expectations: float32 NHWC normalized sample data.
    data = np.random.random((20, 128, 128, 3)).astype(np.float32)
    np.save(sample_path, data)
    print(f"Created local onnx2tf sample data file: {sample_path}")


def run_quantization_phase(onnx_path: str, output_dir: str, calibration_npy: str | None) -> None:
    ensure_onnx2tf_sample_data_file()

    primary_cmd = build_onnx2tf_cmd(
        onnx_path,
        output_dir,
        calibration_npy,
        include_quantization=True,
        include_validation=False,
    )

    try:
        subprocess.run(primary_cmd, check=True)
        return
    except subprocess.CalledProcessError as exc:
        print(f"onnx2tf failed on primary command (exit={exc.returncode}).")

    for candidate in _auto_json_candidates(onnx_path, output_dir):
        if candidate.is_file():
            retry_cmd = primary_cmd + ['-prf', str(candidate)]
            print(f"Retrying onnx2tf using parameter replacement file: {candidate}")
            try:
                subprocess.run(retry_cmd, check=True)
                return
            except subprocess.CalledProcessError as exc:
                print(f"onnx2tf retry with replacement JSON failed (exit={exc.returncode}).")

    if calibration_npy:
        relaxed_cmd = primary_cmd + ['-dsm']
        for candidate in _auto_json_candidates(onnx_path, output_dir):
            if candidate.is_file():
                relaxed_cmd += ['-prf', str(candidate)]
                break
        print("INT8 strict conversion failed. Retrying with -dsm (disable strict mode).")
        try:
            subprocess.run(relaxed_cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            print(f"onnx2tf relaxed retry failed (exit={exc.returncode}).")

        print("INT8 conversion failed after retries. Falling back to float conversion.")
        fallback_cmd = build_onnx2tf_cmd(
            onnx_path,
            output_dir,
            calibration_npy=None,
            include_quantization=False,
            include_validation=False,
        )
        subprocess.run(fallback_cmd, check=True)
        return

    raise RuntimeError("onnx2tf conversion failed. Try running with -agj manually and inspect generated *_auto.json.")


def run_validation_phase(onnx_path: str, output_dir: str) -> None:
    ensure_onnx2tf_sample_data_file()

    validation_cmd = build_onnx2tf_cmd(
        onnx_path,
        output_dir,
        calibration_npy=None,
        include_quantization=False,
        include_validation=True,
    )
    print("Running optional ONNX↔TF validation phase (--validate).")
    try:
        result = subprocess.run(
            validation_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_text = result.stdout or ""
        log_text_lower = log_text.lower()

        if "matches" in log_text_lower:
            verdict = "Matches"
        elif "unmatched" in log_text_lower:
            verdict = "Unmatched"
        elif "skipped" in log_text_lower or "accuracy error measurement process was skipped" in log_text_lower:
            verdict = "Skipped"
        else:
            verdict = "Unknown"

        print(f"Validation phase completed. Result: {verdict}")
        if verdict in {"Unmatched", "Skipped", "Unknown"}:
            relevant_lines = [
                line
                for line in log_text.splitlines()
                if "Matches" in line
                or "Unmatched" in line
                or "Skipped" in line
                or "accuracy error measurement process was skipped" in line
                or "INVALID_ARGUMENT" in line
            ]
            if relevant_lines:
                print("Validation details:")
                for line in relevant_lines[-8:]:
                    print(line)
            elif verdict == "Unknown":
                print("Validation details (tail):")
                for line in log_text.splitlines()[-12:]:
                    print(line)
    except subprocess.CalledProcessError as exc:
        print(f"Validation phase failed (exit={exc.returncode}). Conversion artifacts are still available.")


def export_to_tflite(model, dummy_input, validate: bool = False):
    # Convert the PyTorch model to ONNX format
    onnx_path = str(MODELS_PATH / f'{MODEL_PREFIX}.onnx')
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True)

    # Convert the ONNX model to TFLite via onnx2tf
    onnx2tf_output_dir = str(MODELS_PATH / 'saved_model')
    os.makedirs(onnx2tf_output_dir, exist_ok=True)

    calibration_npy = get_calibration_npy()
    run_quantization_phase(onnx_path, onnx2tf_output_dir, calibration_npy)

    if validate:
        run_validation_phase(onnx_path, onnx2tf_output_dir)


# ==================================Main============================================ #

def main():
    parser = argparse.ArgumentParser(description='PyTorch -> ONNX -> TFLite export pipeline.')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run an additional ONNX↔TF validation phase after conversion.',
    )
    args = parser.parse_args()

    # model = XOR_model().to(device)
    model = NNmodel().to(device)
    model.eval()
 
    dummy_input = torch.randn(*DUMMY_INPUT_SHAPE, device=device)

    summary(model, input_size=DUMMY_INPUT_SHAPE)
    export_to_tflite(model, dummy_input, validate=args.validate)

if __name__ == '__main__':
    main()