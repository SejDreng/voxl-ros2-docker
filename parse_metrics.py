import re
import numpy as np
import sys

def parse_file(filepath):
    print(f"\nProcessing: {filepath}")
    frames = []
    fps_list = []
    latency_list = []
    pre_list = []
    infer_list = []
    post_list = []
    
    pattern = re.compile(r"Frame=(\d+) \| FPS=([\d.]+) \| Latency=([\d.]+)ms \(preprocess=([\d.]+)ms, infer=([\d.]+)ms, postprocess=([\d.]+)ms\)")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    parsed_lines = []
    for line in lines:
        match = pattern.search(line)
        if match:
            parsed_lines.append(line.strip())
            f_idx, fps, lat, pre, inf, post = map(float, match.groups())
            fps_list.append(fps)
            latency_list.append(lat)
            pre_list.append(pre)
            infer_list.append(inf)
            post_list.append(post)

    if not fps_list:
        print("No matches found.")
        return

    print("First 3 lines:")
    for l in parsed_lines[:3]: print(f"  {l}")
    print("Last 3 lines:")
    for l in parsed_lines[-3:]: print(f"  {l}")

    print(f"\nSummary for {filepath.split('/')[-2]}:")
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 30)
    print(f"{'Frames':<15} | {len(fps_list)}")
    print(f"{'Mean FPS':<15} | {np.mean(fps_list):.2f}")
    print(f"{'Median FPS':<15} | {np.median(fps_list):.2f}")
    print(f"{'Mean Latency':<15} | {np.mean(latency_list):.2f} ms")
    print(f"{'P50 Latency':<15} | {np.percentile(latency_list, 50):.2f} ms")
    print(f"{'P95 Latency':<15} | {np.percentile(latency_list, 95):.2f} ms")
    print(f"{'Mean Pre':<15} | {np.mean(pre_list):.2f} ms")
    print(f"{'Mean Infer':<15} | {np.mean(infer_list):.2f} ms")
    print(f"{'Mean Post':<15} | {np.mean(post_list):.2f} ms")

files = [
    "/home/adrian/Git_Repositories/voxl-ros2-docker-applications/logs/yolo8n_pytorch_dev_CPU/metrics_20260408_103418.log",
    "/home/adrian/Git_Repositories/voxl-ros2-docker-applications/logs/yolo8n_pytorch_dev_GPU/metrics_20260408_110117.log"
]

for f in files:
    parse_file(f)
