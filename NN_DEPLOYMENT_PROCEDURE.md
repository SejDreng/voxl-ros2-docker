# Neural Network Deployment Procedure (VOXL2 + ROS2)

This document describes a complete workflow for training, converting, and deploying ROS2 inference nodes on VOXL2 using:

1. **TensorFlow Lite runtime** (primary, via PyTorch → ONNX → onnx2tf conversion)
2. **Native PyTorch runtime** (fallback, for CPU-only inference)
3. **ROS2 Python nodes** with flexible backend selection

Supports multiple model architectures: YOLO object detection, regression models, classification, segmentation, depth estimation, and custom models.

---

## Scope and Assumptions

- **Workstation**: x86_64 Linux with Docker (for cross-compilation to ARM64)
- **Drone**: VOXL2 (aarch64) with Docker runtime
- **Repository root**: `/path/to/workspace/voxl-ros2-docker`
- **Model types supported**: TorchScript (`.pt`), checkpoint dicts, PyTorch modules
- **Input shapes**: dynamic (regression: `(1, N)`; detection: `(1, 3, H, W)`; custom: any)
- **Inference platforms**: TFLite (recommended) or PyTorch

---

## Part 1: Model Conversion Pipeline

### 1.1 Prepare Your Model

Place your trained PyTorch model at:
```
models/<your-model-name>.pt
```

Examples:
- `models/best_model_20260319-141220_dp.pt` (regression)
- `models/yolov8n.pt` (object detection)
- `models/custom_classifier.pt` (classification)

**Model format**: Supports TorchScript (JIT-compiled), checkpoint dicts, or saved nn.Module instances.

### 1.2 Run the Dynamic Conversion Pipeline

The conversion script automatically detects model type and input shape. Core features:

- **Automatic TorchScript handling** (no manual tracing required)
- **Dynamic input shape support** (via `--input-shape` CLI argument)
- **Skip calibration option** (for fast float32 export)
- **Optional quantization validation** (INT8 via onnx2tf)
- **Fallback retry logic** (INT8 strict → relaxed → float conversion)

#### Basic usage (regression model, shape 1,16):

```bash
cd /home/adrian/Git_Repositories/voxl-ros2-docker

# Float32 TFLite only (no quantization):
python3 scripts/pt_onnx_tflite_pipeline.py --skip-quantization

# With calibration and INT8 quantization:
python3 scripts/pt_onnx_tflite_pipeline.py
```

#### Custom model with different input shape:

```bash
# YOLO model (1, 3, 320, 320):
python3 scripts/pt_onnx_tflite_pipeline.py \
  --model models/yolov8n.pt \
  --input-shape 1,3,320,320 \
  --skip-quantization

# Classification model (1, 3, 224, 224):
python3 scripts/pt_onnx_tflite_pipeline.py \
  --model models/classifier.pt \
  --input-shape 1,3,224,224
```

#### With validation:

```bash
python3 scripts/pt_onnx_tflite_pipeline.py \
  --input-shape 1,16 \
  --skip-quantization \
  --validate
```

#### Script arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `models/best_model_20260319-141220_dp.pt` | Path to PyTorch model file |
| `--input-shape` | `1,16` | Input dimensions (comma-separated, e.g., `1,3,224,224`) |
| `--skip-quantization` | False | Export float32 and float16 only; skip calibration and INT8 |
| `--validate` | False | Run ONNX↔TensorFlow validation phase |

### 1.3 Conversion Outputs

Upon successful conversion, three artifacts are generated in `models/`:

```
models/
├── <model-name>.onnx                    # ONNX intermediate
└── saved_model/
    ├── <model-name>_float32.tflite      # Full precision (always generated)
    ├── <model-name>_float16.tflite      # Half precision (always generated)
    ├── <model-name>_dynamic_range_quant.tflite    # INT8 (if quantization enabled)
    ├── <model-name>_full_integer_quant.tflite     # Full INT8 (if quantization enabled)
    └── ...
```

**For regression** (small models): `_float32.tflite` is recommended.

### 1.4 Calibration Data (Optional)

For CV quantization, place calibration images under:
```
models/quantization_data/Images/
```

Supported formats: `*.jpg`, `*.jpeg`, `*.png`, `*.bmp`, `*.webp`

**Image requirements**: 
- Should represent typical input distribution
- At least 10–100 images recommended
- Automatically resized to match model input dimensions

If no calibration data is found, quantization falls back to dynamic range quantization or float export. 

If calibrating a regression model, create a .npy file named `calibration_data.npy` containing input values for the model dimensions and place it at `models/quantization_data`. The script checks for this file before it creates one from provided images. 

---


### 3.4 Logs and plotting


```bash
# View real-time logs from VOXL
make voxl-logs

# SSH into VOXL and inspect ROS topics
ros2 topic list
ros2 topic echo /xor_output

# Check system resource usage inside container
docker stats voxl-runtime
```

```
---

## Part 5: Troubleshooting

### Issue: Model not found on VOXL

**Solution**: Verify model path is absolute and mounted in container:
```bash
docker exec voxl-runtime ls -la /ros2_ws/src/nn_inference_node/nn_inference_node/models/
```

### Issue: TFLite inference very slow or hanging

**Solution**: 
- Check if model is too large for available RAM
- Try float32 model instead of quantized
- Verify input shape matches model expectations

### Issue: Import errors for tflite_runtime or torch

**Solution**: Reinstall dependencies in container:
```bash
make voxl-run "pip install tflite-runtime torch"
```

### Issue: ONNX export fails with "Unsupported op"

**Solution**: 
- Try lower opset version (script defaults to 12)
- Use `--validate` flag to see detailed conversion logs
- Simplify model (remove custom layers if possible)

---

## Part 6: Quick Reference

| Task | Command |
|------|---------|
| Convert regression model (1,16) | `python3 scripts/pt_onnx_tflite_pipeline.py --skip-quantization` |
| Convert YOLO (1,3,320,320) | `python3 scripts/pt_onnx_tflite_pipeline.py --model models/yolov8n.pt --input-shape 1,3,320,320` |
| Build workspace (cross) | `make build-cross && make build-ws-cross` |
| Deploy to VOXL | `make deploy && make deploy-image` |
| Start VOXL container | `make voxl-start` |
| View logs | `make voxl-logs` |
| Run inference node | `ros2 run nn_inference_node node` |
| Stop VOXL | `make voxl-stop` |

---

## References

- [PyTorch to ONNX export](https://pytorch.org/docs/stable/onnx.html)
- [onnx2tf GitHub](https://github.com/PINTO0309/onnx2tf)
- [TensorFlow Lite Python API](https://www.tensorflow.org/lite/guide/python)
- [VOXL2 documentation](https://docs.modalai.com/voxl2-reference/)
