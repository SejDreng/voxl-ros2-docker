import torch
import torch.nn as nn
from torch import Tensor
from torchinfo import summary

import os
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np


if torch.cuda.is_available():
    device = torch.device('cuda')
else:    
    device = torch.device('cpu')

# ====================================Path Configuration========================================== #


# # REPLACE WITH YOUR OWN MODEL NAME
# MODEL_NAME = 'best_model_20260319-141220_dp.pt'
# # MODEL_NAME = 'yolov8n.pt'
# MODEL_PREFIX = Path(MODEL_NAME).stem
# # DUMMY_INPUT_SHAPE = (1, 3, 640, 640)  # Adjust as needed for your model
# DUMMY_INPUT_SHAPE = (1, 16)  # Adjust as needed for your model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / 'models'

CALIBRATION_DATA_PATH = MODELS_PATH / 'quantization_data' / 'Images'
CALIBRATION_NPY_PATH = MODELS_PATH / 'quantization_data' / 'calibration_data.npy'

# Default model (can be overridden via --model CLI arg)
DEFAULT_MODEL_NAME = 'best_model_20260319-141220_dp.pt'
DEFAULT_INPUT_SHAPE = (1, 16)


# ==================================Specify Model============================================ #

class NNmodel(nn.Module):
    def __init__(self, model_path: str):
        super(NNmodel, self).__init__()
        self.model, self.is_torchscript = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str) -> tuple[torch.nn.Module, bool]:
        """Load either a checkpoint dict, a regular module, or a TorchScript archive."""
        loaded = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(loaded, dict) and 'model' in loaded:
            return loaded['model'].float().to(device), False

        if isinstance(loaded, torch.jit.ScriptModule):
            return loaded, True

        return loaded, False

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
# ==================================Export Pipeline============================================ #


def resolve_model_path(model_arg: str | None) -> Path:
    """Resolve the model path relative to the models directory when needed."""
    if model_arg is None:
        return MODELS_PATH / DEFAULT_MODEL_NAME

    model_path = Path(model_arg)
    if not model_path.is_absolute():
        model_path = MODELS_PATH / model_path
    return model_path


def parse_input_shape(shape_arg: str | None) -> tuple[int, ...]:
    """Parse a comma-separated input shape string into a tuple of ints."""
    if shape_arg is None:
        return DEFAULT_INPUT_SHAPE

    try:
        return tuple(int(value.strip()) for value in shape_arg.split(','))
    except ValueError as exc:
        raise ValueError(
            f"Invalid input shape: {shape_arg}. Expected comma-separated integers."
        ) from exc
    
def get_calibration_npy(input_shape: tuple) -> str | None:
    # For image inputs (4D), extract H and W; for other shapes, skip calibration
    if len(input_shape) < 4:
        print(f"Input shape {input_shape} is not image-like (need 4D NCHW). Skipping calibration.")
        return None
    
    expected_h = int(input_shape[2])
    expected_w = int(input_shape[3])

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


def export_to_tflite(model, dummy_input, validate: bool = False, skip_quantization: bool = False, model_prefix: str = 'model'):
    # Convert the PyTorch model to ONNX format
    onnx_path = str(MODELS_PATH / f'{model_prefix}.onnx')
    
    # Check if model is a TorchScript model and export it appropriately
    if model.is_torchscript:
        print("Detected TorchScript model. Exporting to ONNX using the legacy exporter...")
        torch.onnx.utils.export(
            model.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            verbose=False,
            input_names=['input'],
            output_names=['output']
        )
    else:
        torch.onnx.export(model, dummy_input, onnx_path, export_params=True)

    # Convert the ONNX model to TFLite via onnx2tf
    onnx2tf_output_dir = str(MODELS_PATH / 'saved_model')
    os.makedirs(onnx2tf_output_dir, exist_ok=True)

    if skip_quantization:
        print("Skipping quantization phase (--skip-quantization flag set).")
        # Run float32 conversion without calibration
        ensure_onnx2tf_sample_data_file()
        float_cmd = build_onnx2tf_cmd(
            onnx_path,
            onnx2tf_output_dir,
            calibration_npy=None,
            include_quantization=False,
            include_validation=False,
        )
        subprocess.run(float_cmd, check=True)
    else:
        calibration_npy = get_calibration_npy(dummy_input.shape)
        run_quantization_phase(onnx_path, onnx2tf_output_dir, calibration_npy)

    if validate:
        run_validation_phase(onnx_path, onnx2tf_output_dir)


# ==================================Main============================================ #

def main():
    parser = argparse.ArgumentParser(description='PyTorch -> ONNX -> TFLite export pipeline.')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help=f'Path to model file (default: {MODELS_PATH / DEFAULT_MODEL_NAME})',
    )
    parser.add_argument(
        '--input-shape',
        type=str,
        default=None,
        help=f'Input shape as comma-separated integers (default: {DEFAULT_INPUT_SHAPE}). Example: 1,16 or 1,3,224,224',
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run an additional ONNX↔TF validation phase after conversion.',
    )
    parser.add_argument(
        '--skip-quantization',
        action='store_true',
        help='Skip calibration data phase and export as float32 TFLite (no quantization).',
    )
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_prefix = model_path.stem

    input_shape = parse_input_shape(args.input_shape)
    
    print(f"Model: {model_path}")
    print(f"Input shape: {input_shape}")

    model = NNmodel(str(model_path)).to(device)
    model.eval()
 
    dummy_input = torch.randn(*input_shape, device=device)

    summary(model, input_size=input_shape)
    export_to_tflite(model, dummy_input, validate=args.validate, skip_quantization=args.skip_quantization, model_prefix=model_prefix)

if __name__ == '__main__':
    main()