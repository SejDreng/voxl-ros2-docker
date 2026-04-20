# VOXL-ROS2 libmodal-pipe Integration

## Overview

The `voxl_modal_pipe_reader_node` reads YOLOv8 detections directly from **voxl-tflite-server** using libmodal-pipe (v2.13.2+) and converts them to the standardized JSON contract.

## Architecture

### Data Flow

```
VOXL Host (Ubuntu 18.04)
├─ voxl-tflite-server (process)
│  └─ Outputs binary detections → /run/mpa/tflite (named pipe)
│
└─ Docker container (Ubuntu 22.04 + ROS2 Humble)
   ├─ voxl_modal_pipe_reader_node (C++, libmodal-pipe client)
   │  └─ Reads from /run/mpa/tflite → Parses YOLOv8 binary → JSON
   │
   └─ ROS2 pub/sub
      └─ Publishes JSON on /tflite_server/detections_raw
```

### Binary Format

voxl-tflite-server outputs YOLOv8 detections as fixed-size binary packets:

```c
struct Detection {
  float x_center;    // Bounding box center X coordinate
  float y_center;    // Bounding box center Y coordinate
  float width;       // Bounding box width
  float height;      // Bounding box height
  float confidence;  // Detection confidence [0, 1]
  int   class_id;    // COCO class ID (0-79)
  float reserved;    // Padding/reserved for future use
} __packed;          // Total: 28 bytes per detection
```

The reader validates:
- `confidence >= confidence_threshold` (default: 0.5)
- `0 <= confidence <= 1.0`
- `width > 0, height > 0`

### Coordinate System

Detections are converted from **center-based** to **corner-based**:

```
Input:  (x_center, y_center, width, height)
Output: (x_min, y_min, x_max, y_max)
  where:
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
```

## Building on Workstation

The reader node requires libmodal-pipe headers, which are **only available on VOXL**. To cross-compile:

### Option 1: Build in Docker (Recommended)

The multi-stage Dockerfile includes libmodal-pipe in the `voxl-deps` stage. Build normally:

```bash
make build-ws
```

This compiles the voxl_tflite_bridge package with libmodal-pipe linked in.

### Option 2: Manual Cross-Compilation

If building outside Docker:

```bash
# Install libmodal-pipe development headers (from VOXL)
# Copy or install: libmodal-pipe-dev (packages) and modal_pipe*.h headers
# Either via scp, apt-get on VOXL, or from ModalAI SDK

# Then in your colcon workspace:
colcon build --packages-select voxl_tflite_bridge
```

## Running on VOXL

### Prerequisites

1. **voxl-tflite-server running** with models at `/usr/bin/dnn/`
   ```bash
   voxl2:~$ cat /etc/modalai/voxl-tflite-server.conf
   {
     "model": "/usr/bin/dnn/yolov8n_float32.tflite",
     "input_pipe": "hires",
     "delegate": "gpu",
     ...
   }
   ```

2. **libmodal-pipe v2.13.2+**
   ```bash
   voxl2:~$ dpkg -l | grep libmodal-pipe
   ii  libmodal-pipe  2.13.2  arm64
   ```

3. **ROS2 node deployed** (see [NN_DEPLOYMENT_PROCEDURE.md](../../NN_DEPLOYMENT_PROCEDURE.md))

### Launch

**On VOXL (in ROS2 container or native):**

```bash
ros2 launch voxl_tflite_bridge voxl_modal_pipe_reader.launch.py \
  model_name:=yolov8n_float32 \
  camera_name:=hires \
  confidence_threshold:=0.5 \
  output_topic:=/tflite_server/detections_raw
```

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `output_topic` | str | `/tflite_server/detections_raw` | ROS topic for JSON detections |
| `model_name` | str | `yolov8n_full_integer_quant.tflite` | Model filename (for metadata) |
| `camera_name` | str | `hires` | Camera source name (for metadata) |
| `confidence_threshold` | float | `0.5` | Min confidence to publish detection |

### Troubleshooting

**"Failed to open pipe 'tflite'. Is voxl-tflite-server running?"**
- Verify: `systemctl status voxl-tflite-server`
- Check config: `cat /etc/modalai/voxl-tflite-server.conf`
- Restart: `sudo systemctl restart voxl-tflite-server`

**"Received X bytes, not a multiple of 28"**
- The pipe is returning data in an unexpected format
- Check if voxl-tflite-server version changed or has different output format
- Debug: Run with `ROS_LOG_LEVEL=DEBUG` to see individual raw packets

**Detections not publishing**
- Check `/tflite_server/detections_raw` topic: `ros2 topic echo /tflite_server/detections_raw`
- Verify confidence threshold: Some models have naturally low confidence scores
- Check ROS connectivity: `ros2 topic list` should show the topic active

## JSON Output Format

Once validated by `voxl_tflite_bridge_node`, detections are republished on `/tflite_server/detections`:

```json
{
  "meta": {
    "timestamp_ns": 1712956800000000000,
    "frame_id": 42,
    "camera": "hires",
    "model": "yolov8n_float32"
  },
  "detections": [
    {
      "x_min": 100,
      "y_min": 200,
      "x_max": 350,
      "y_max": 450,
      "score": 0.92,
      "class": 0
    }
  ]
}
```

## Performance Considerations

- **Pipe buffer size**: 64KB (can hold ~2300 detections at 7 floats each)
- **Callback overhead**: Minimal; libmodal-pipe handles threading
- **JSON serialization**: Happens in ROS node, not in reader
- **Confidence filtering**: Applied in binary domain before JSON serialization

## API References

- libmodal-pipe: `/usr/include/modal_pipe_client.h` (on VOXL)
- voxl-tflite-server: v0.4.1 (checked via `dpkg -l`)
- YOLO format: YOLOv8 nano, outputs 84 features per detection (4 bbox + 80 class logits)
