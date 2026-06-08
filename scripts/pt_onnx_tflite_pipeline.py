import argparse
import onnx2tf
import timm
import numpy as np
from pathlib import Path
from ultralytics import RTDETR, YOLO

import torch
import torch.nn as nn
from torch import Tensor
from torchinfo import summary
from collections import OrderedDict


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / 'models'
DEFAULT_MODEL_NAME = 'xor_model.pt'
DEFAULT_INPUT_SHAPE = (1, 2)
DEFAULT_ONNX_OPSET = 18
CALIBRATION_SAMPLES = 100


class NNModel(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.model, self.is_torchscript = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str) -> tuple[torch.nn.Module, bool]:
        try:
            scripted_model = torch.jit.load(model_path, map_location=DEVICE)
            return scripted_model, True
        except Exception:
            pass

        loaded = torch.load(model_path, map_location=DEVICE, weights_only=False)

        if isinstance(loaded, dict) and 'model' in loaded:
            return loaded['model'].float().to(DEVICE), False

        if isinstance(loaded, torch.jit.ScriptModule):
            return loaded, True

        if isinstance(loaded, nn.Module):
            return loaded, False

        if isinstance(loaded, (dict, OrderedDict)):
            # Loaded is a state_dict — requires known architecture
            model_name = 'mobilevitv2_100'
            model = timm.create_model(model_name, pretrained=False)
            model.load_state_dict(loaded)
            model.to(DEVICE)
            model.eval()
            return model, False

        raise TypeError(
            f'Unsupported model type from {model_path}: {type(loaded).__name__}. '
            'Expected checkpoint dict with "model", nn.Module, or TorchScript.'
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def resolve_model_path(model_arg: str | None) -> Path:
    if model_arg is None:
        return MODELS_PATH / DEFAULT_MODEL_NAME

    model_path = Path(model_arg)
    if not model_path.is_absolute():
        model_path = MODELS_PATH / model_path
    return model_path


def parse_input_shape(shape_arg: str | None) -> tuple[int, ...]:
    if shape_arg is None:
        return DEFAULT_INPUT_SHAPE

    try:
        return tuple(int(value.strip()) for value in shape_arg.split(','))
    except ValueError as exc:
        raise ValueError(
            f'Invalid input shape: {shape_arg}. Expected comma-separated integers.'
        ) from exc


def export_to_onnx(model: NNModel, dummy_input: Tensor, model_prefix: str, onnx_opset: int) -> Path:
    onnx_path = MODELS_PATH / f'{model_prefix}.onnx'

    export_model = model.model if model.is_torchscript else model
    export_model.eval()

    torch.onnx.export(
        export_model,
        (dummy_input,),
        str(onnx_path),
        export_params=True,
        opset_version=onnx_opset,
        do_constant_folding=True,
        verbose=False,
        input_names=['inputs_0'],
        output_names=['Identity'],
        dynamo=False if model.is_torchscript else True,
    )

    return onnx_path


def make_calib_data(input_shape: tuple[int, ...], n: int = CALIBRATION_SAMPLES) -> Path:
    calib_path = MODELS_PATH / 'calib_data.npy'
    # input_shape is NCHW from torch (1, 3, H, W) — save as-is for onnxruntime
    data = np.random.rand(n, *input_shape[1:]).astype(np.float32)
    np.save(str(calib_path), data)
    return calib_path


def convert_onnx_to_tflite(
    onnx_path: Path,
    output_dir: Path,
    input_shape: tuple[int, ...],
    validate: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    is_image_input = len(input_shape) == 4
    calib_path = make_calib_data(input_shape)

    if is_image_input:
        calib_mean = '[[[[0.0]], [[0.0]], [[0.0]]]]'
        calib_std  = '[[[[1.0]], [[1.0]], [[1.0]]]]'
    else:
        calib_mean = '[[0.0]]'
        calib_std  = '[[1.0]]'

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_dir),
        # only pass for image models — 2D inputs don't have a channel layout to preserve
        **({"keep_ncw_or_nchw_or_ncdhw_input_names": ["inputs_0"]} if is_image_input else {}),
        output_dynamic_range_quantized_tflite=True,
        output_integer_quantized_tflite=True,
        disable_suppression_flexstridedslice=True,
        number_of_dimensions_after_flexstridedslice_compression=10,
        quant_type='per-channel',
        input_quant_dtype='int8',
        output_quant_dtype='int8',
        quant_norm_mean=calib_mean,
        quant_norm_std=calib_std,
        custom_input_op_name_np_data_path=[
            ['inputs_0', str(calib_path), 0.0, 1.0]
        ],
        copy_onnx_input_output_names_to_tflite=validate,
        check_onnx_tf_outputs_elementwise_close=validate,
        check_onnx_tf_outputs_elementwise_close_full=validate,
        auto_generate_json=True,
        auto_generate_json_on_error=True,
        non_verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Regression model: PyTorch -> ONNX -> TFLite export.')
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
        help=f'Input shape as comma-separated integers (default: {DEFAULT_INPUT_SHAPE}). Example: 1,16',
    )
    parser.add_argument(
        '--onnx-opset',
        type=int,
        default=DEFAULT_ONNX_OPSET,
        help=f'ONNX opset version for torch export (default: {DEFAULT_ONNX_OPSET}).',
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run extra ONNX<->TF validation in onnx2tf.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(MODELS_PATH / 'saved_model'),
        help='Directory where onnx2tf writes TFLite artifacts.',
    )
    parser.add_argument(
        '--YOLO',
        action='store_true',
        help='Use YOLO model.',
    )
    parser.add_argument(
        '--RTDETR',
        action='store_true',
        help='Use RT-DETR model.',
    )
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f'Model file not found: {model_path}')

    input_shape = parse_input_shape(args.input_shape)
    model_prefix = model_path.stem

    print(f'Model:       {model_path}')
    print(f'Input shape: {input_shape}')

    if args.YOLO or args.RTDETR:
        if args.YOLO and args.RTDETR:
            raise ValueError('Choose only one detector type: --YOLO or --RTDETR.')
        if args.YOLO:
            model = YOLO(str(model_path))
            model.export(
                format='tflite',
                imgsz=input_shape[2],
                int8=True,
            )
        else:
            model = RTDETR(str(model_path))
            model.export(
                format='tflite',
                imgsz=input_shape[2],
                simplify=True,
                nms=False,
            )
        return

    model = NNModel(str(model_path)).to(DEVICE)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=DEVICE)
    summary(model, input_size=input_shape)

    onnx_path = export_to_onnx(model, dummy_input, model_prefix, args.onnx_opset)
    print(f'ONNX export complete:   {onnx_path}')

    output_dir = Path(args.output_dir)
    convert_onnx_to_tflite(onnx_path, output_dir, input_shape, validate=args.validate)
    print(f'TFLite conversion complete: {output_dir}')


if __name__ == '__main__':
    main()