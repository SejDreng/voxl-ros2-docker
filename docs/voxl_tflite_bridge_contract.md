# VOXL TFLite Bridge Contract

This repository's Python node no longer runs the TFLite model itself.
Instead, it expects a VOXL-side bridge to publish detections as JSON on a ROS topic.

The buildable ROS2 bridge package added in this workspace lives at:

- [ros2_ws/src/voxl_tflite_bridge](../ros2_ws/src/voxl_tflite_bridge)

## Topics

- Input image topic: `/hires_small_color` by default
- Preprocessed image topic: `/nn_inference/preprocessed_image`
- Detections topic: `/tflite_server/detections`
- Annotated output topic: `/nn_inference/output_image`

## Detection message format

The bridge should publish a `std_msgs/String` containing JSON with this shape:

```json
{
  "meta": {
    "timestamp_ns": 1712928000000000000,
    "frame_id": 123,
    "camera": "hires",
    "model": "yolov8n_full_integer_quant.tflite"
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "class_confidence": 0.98,
      "detection_confidence": 0.94,
      "x_min": 42.0,
      "y_min": 31.0,
      "x_max": 188.0,
      "y_max": 401.0
    }
  ]
}
```

## Required fields

- `meta.timestamp_ns` or `meta.frame_id` so the Python node can match detections back to a cached frame.
- `detections[]` array.
- For each detection, box coordinates plus confidence values.

## Accepted aliases

The Python node accepts either of these box key sets:

- `x_min`, `y_min`, `x_max`, `y_max`
- `x1`, `y1`, `x2`, `y2`

It also accepts `score` as a fallback for `class_confidence` and `detection_confidence`.

## Behavior

- The Python node scales detections from the preprocessed space to the original image space unless `NN_DETECTION_SPACE=original`.
- Non-JSON or malformed payloads are ignored.
- The node limits NMS and max detections locally so the bridge can stay simple.

## Build and launch

```bash
colcon build --packages-select voxl_tflite_bridge
source install/setup.bash
ros2 launch voxl_tflite_bridge voxl_tflite_bridge.launch.py
ros2 launch voxl_tflite_bridge voxl_modal_pipe_reader.launch.py
```

The bridge package is intentionally lightweight: it validates and forwards the JSON payload.
The actual libmodal-pipe reader still belongs on the VOXL side, where it can convert the tflite-server pipe output into the JSON contract above.

A source entrypoint for that reader lives at:

- [ros2_ws/src/voxl_tflite_bridge/src/voxl_modal_pipe_reader_node.cpp](../ros2_ws/src/voxl_tflite_bridge/src/voxl_modal_pipe_reader_node.cpp)
