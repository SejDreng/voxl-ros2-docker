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
import subprocess
import sys
from pathlib import Path

import tensorflow as tf


if torch.cuda.is_available():
    device = torch.device('cuda')
else:    
    device = torch.device('cpu')

# ====================================Path Configuration========================================== #


# REPLACE WITH YOUR OWN MODEL NAME
MODEL_NAME = 'yolov8n.pt'
DUMMY_INPUT_SHAPE = (1, 3, 640, 640)  # Adjust as needed for your model

# REPLACE WITH YOUR OWN CALIBRATION DATA PATH
CALIBRATION_DATA_PATH = './quantization_data/Images'
CALIBRATION_NPY_PATH = './quantization_data/calibration_data.npy'

MODELS_PATH = './models/'
TFLITE_MODELS_PATH = './models/saved_models/'


# ==================================Specify Model============================================ #

class NNmodel(nn.Module):
    def __init__(self):
        super(NNmodel, self).__init__()
        
        # yolo = YOLO('yolov8n.pt')  # downloads weights if not cached
        # self.model = yolo.model.float().to(device)
        
        # alternative, load yolo model from local file
        model_path = os.path.join(MODELS_PATH, MODEL_NAME)
        model = torch.load(model_path, map_location=device, weights_only=False)
        self.model = model['model'].float().to(device)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
# ==================================Export Pipeline============================================ #
    
def get_calibration_npy() -> str | None:
    if os.path.isfile(CALIBRATION_NPY_PATH):
        return CALIBRATION_NPY_PATH

    image_patterns = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    calibration_dir = Path(CALIBRATION_DATA_PATH)
    if calibration_dir.is_dir() and any(calibration_dir.rglob(p) for p in image_patterns):
        calibration_script = os.path.join(os.path.dirname(__file__), 'build_calibration_npy.py')
        subprocess.run(
            [sys.executable, calibration_script,
             '--input-dir', CALIBRATION_DATA_PATH,
             '--output', CALIBRATION_NPY_PATH,
             '--recursive'],
            check=True,
        )
        return CALIBRATION_NPY_PATH

    return None


def build_onnx2tf_cmd(onnx_path: str, output_dir: str, calibration_npy: str | None) -> list[str]:
    cmd = ['onnx2tf', '-i', onnx_path, '-o', output_dir]
    if calibration_npy:
        print(f"Using INT8 calibration data: {calibration_npy}")
        cmd += ['-oiqt', '-cind', 'x', calibration_npy, '-iqd', 'int8', '-oqd', 'int8']
    else:
        print("No calibration data found. Running non-INT8 onnx2tf conversion.")
    return cmd


def export_to_tflite(model, dummy_input):
    # Convert the PyTorch model to ONNX format
    onnx_path = os.path.join(MODELS_PATH, 'yolov8n.onnx')
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True)

    # Convert the ONNX model to TFLite via onnx2tf
    onnx2tf_output_dir = os.path.join(MODELS_PATH, 'saved_model')
    os.makedirs(onnx2tf_output_dir, exist_ok=True)

    calibration_npy = get_calibration_npy()
    cmd = build_onnx2tf_cmd(onnx_path, onnx2tf_output_dir, calibration_npy)
    subprocess.run(cmd, check=True)


# ==================================Main============================================ #

def main():
    # model = XOR_model().to(device)
    model = NNmodel().to(device)
    model.eval()
 
    dummy_input = torch.randn(*DUMMY_INPUT_SHAPE, device=device)

    summary(model, input_size=DUMMY_INPUT_SHAPE)
    export_to_tflite(model, dummy_input)

if __name__ == '__main__':
    main()