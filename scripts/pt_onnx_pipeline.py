import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
import torchvision
import torch.onnx

import os
from ament_index_python.packages import get_package_share_directory


if torch.cuda.is_available():
    device = torch.device('cuda')
else:    
    device = torch.device('cpu')
    

# Standard initialization of a custom model
# class XOR_model(nn.Module):
#     def __init__(self):
#         super(XOR_model, self).__init__()
#         self.linear1 = nn.Linear(2, 8)
#         self.linear15 = nn.Linear(8, 2)
#         self.linear2 = nn.Linear(2, 1)

#     def forward(self, x: Tensor) -> Tensor:
#         x = torch.relu(self.linear1(x))
#         x = torch.relu(self.linear15(x))
#         x = torch.sigmoid(self.linear2(x))
#         return x

class yolo8n_model(nn.Module):
    def __init__(self):
        super(yolo8n_model, self).__init__()
        # self.model = torch.hub.load('ultralytics/yolov8', 'yolov8n.pt')
        
        # alternative, load yolo model from local file
        package_share_directory = get_package_share_directory('voxl_ros2_pipeline')
        model_path = os.path.join(package_share_directory, 'models', 'yolo8n.pt')
        self.model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    

def export_to_onnx(model, dummy_input, path):
    torch.onnx.export(model, dummy_input, path, opset_version=11, input_names=['input'], output_names=['output'])
    
def export_to_tflite(model, dummy_input):
    # Convert the PyTorch model to ONNX format
    onnx_path = model.path.replace('.tflite', '.onnx')
    export_to_onnx(model, dummy_input, onnx_path)

    # Convert the ONNX model to TensorFlow format using onnx-tf
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(path.replace('.tflite', '.pb'))

    # Convert the TensorFlow model to TensorFlow Lite format using TFLite Converter
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(path.replace('.tflite', '.pb'))
    tflite_model = converter.convert()

    with open(path, 'wb') as f:
        f.write(tflite_model)
        

def main():
    # model = XOR_model().to(device)
    model = yolo8n_model().to(device)
    model.eval()

    # dummy_input = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
    dummy_input = torch.randn(1, 3, 640, 640, device=device)

    export_to_tflite(model, dummy_input, )

if __name__ == '__main__':
    main()