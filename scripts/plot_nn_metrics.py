#!/usr/bin/env python3
"""Plot nn_inference_node metrics from CSV logs.

Usage:
    python3 scripts/plot_nn_metrics.py --csv /ros2_ws/log/nn_inference_logs/metrics_YYYYMMDD_HHMMSS.csv
    python3 scripts/plot_nn_metrics.py --logs-dir /ros2_ws/log/nn_inference_logs
    python3 scripts/plot_nn_metrics.py REGRESSION --logs-dir /ros2_ws/log/nn_XOR_logs

Outputs PNG plots and summary JSON in the chosen output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime hint
    raise SystemExit(
        "matplotlib is required for plotting. Install it with: python3 -m pip install matplotlib"
    ) from exc


@dataclass
class MetricsRow:
    frame: int
    fps: float
    latency_ms: float
    preprocess_ms: float
    infer_ms: float
    postprocess_ms: float
    detections: int
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    avg_bbox_area: float
    timestamp: str
    classes: dict[str, int]


@dataclass
class PlotConfig:
    regression_only: bool
    default_logs_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot nn_inference_node metrics CSV files.")
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help=(
            "Optional positional mode or CSV/directory path. Use REGRESSION to plot only "
            "preprocess/infer/postprocess timing series for regression logs."
        ),
    )
    parser.add_argument("--csv", dest="csv_path", help="Explicit CSV file path.")
    parser.add_argument("--logs-dir", dest="logs_dir", help="Directory containing metrics_*.csv files.")
    parser.add_argument("--outdir", default=None, help="Directory for generated plots and summary JSON. Defaults to <csv_dir>/plots")
    return parser.parse_args()


def plot_config(args: argparse.Namespace) -> PlotConfig:
    regression_only = args.target == "REGRESSION"
    return PlotConfig(
        regression_only=regression_only,
        default_logs_dir="/ros2_ws/log/nn_inference_logs",
    )


def newest_metrics_csv(directory: Path) -> Path:
    candidates = sorted(
        directory.rglob("metrics_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No metrics_*.csv files found in {directory}")
    return candidates[0]


def resolve_csv_path(args: argparse.Namespace, config: PlotConfig) -> Path:
    if args.csv_path:
        return Path(args.csv_path).expanduser().resolve()

    if args.logs_dir:
        directory = Path(args.logs_dir).expanduser().resolve()
        if directory.is_file():
            return directory
        return newest_metrics_csv(directory)

    if args.target and args.target != "REGRESSION":
        path = Path(args.target).expanduser().resolve()
        if path.is_file():
            return path
        if path.is_dir():
            return newest_metrics_csv(path)
        raise FileNotFoundError(f"Path does not exist: {path}")

    path = Path(config.default_logs_dir).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        return newest_metrics_csv(path)
    raise FileNotFoundError(f"Path does not exist: {path}")


def load_rows(csv_path: Path) -> list[MetricsRow]:
    rows: list[MetricsRow] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                rows.append(
                    MetricsRow(
                        frame=int(float(raw.get("frame", 0) or 0)),
                        fps=float(raw.get("fps", 0) or 0),
                        latency_ms=float(raw.get("latency_ms", 0) or 0),
                        preprocess_ms=float(raw.get("preprocess_ms", 0) or 0),
                        infer_ms=float(raw.get("infer_ms", 0) or 0),
                        postprocess_ms=float(raw.get("postprocess_ms", 0) or 0),
                        detections=int(float(raw.get("detections", 0) or 0)),
                        avg_confidence=float(raw.get("avg_confidence", 0) or 0),
                        min_confidence=float(raw.get("min_confidence", 0) or 0),
                        max_confidence=float(raw.get("max_confidence", 0) or 0),
                        avg_bbox_area=float(raw.get("avg_bbox_area", 0) or 0),
                        timestamp=str(raw.get("timestamp", "")),
                        classes=json.loads(raw.get("classes_json", "{}") or "{}"),
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def ensure_outdir(csv_path: Path, outdir: str | None) -> Path:
    candidates: list[Path] = []
    if outdir:
        candidates.append(Path(outdir).expanduser().resolve())
    else:
        candidates.extend([
            csv_path.parent / "plots",
            Path.cwd() / "nn_inference_plots" / csv_path.stem,
            Path.home() / ".cache" / "nn_inference_plots" / csv_path.stem,
            Path("/tmp") / "nn_inference_plots" / csv_path.stem,
        ])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError as exc:
            last_error = exc

    raise PermissionError(
        f"Unable to create any output directory for plots. Last error: {last_error}"
    )


def plot_line(x, series_dict, title, xlabel, ylabel, outpath: Path) -> None:
    plt.figure(figsize=(12, 6))
    for label, values in series_dict.items():
        plt.plot(x, values, label=label, linewidth=1.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if len(series_dict) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_hist(values, title, xlabel, outpath: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=min(30, max(5, len(values) // 4)), color="#3366cc", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def write_summary(rows: list[MetricsRow], csv_path: Path, outdir: Path) -> Path:
    latency_values = [row.latency_ms for row in rows]
    fps_values = [row.fps for row in rows]
    avg_conf_values = [row.avg_confidence for row in rows]
    min_conf_values = [row.min_confidence for row in rows]
    max_conf_values = [row.max_confidence for row in rows]
    summary = {
        "csv_file": str(csv_path),
        "frames": len(rows),
        "mean_fps": round(mean(fps_values), 3),
        "latency_ms": {
            "mean": round(mean(latency_values), 3),
            "p50": round(percentile(latency_values, 50), 3),
            "p95": round(percentile(latency_values, 95), 3),
            "p99": round(percentile(latency_values, 99), 3),
        },
        "preprocess_ms_mean": round(mean(row.preprocess_ms for row in rows), 3),
        "infer_ms_mean": round(mean(row.infer_ms for row in rows), 3),
        "postprocess_ms_mean": round(mean(row.postprocess_ms for row in rows), 3),
        "detections_mean": round(mean(row.detections for row in rows), 3),
        "confidence": {
            "avg_confidence_mean": round(mean(avg_conf_values), 6),
            "avg_confidence_p50": round(percentile(avg_conf_values, 50), 6),
            "avg_confidence_p95": round(percentile(avg_conf_values, 95), 6),
            "min_confidence_mean": round(mean(min_conf_values), 6),
            "max_confidence_mean": round(mean(max_conf_values), 6),
        },
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def write_regression_summary(rows: list[MetricsRow], csv_path: Path, outdir: Path) -> Path:
    summary = {
        "csv_file": str(csv_path),
        "frames": len(rows),
        "preprocess_ms_mean": round(mean(row.preprocess_ms for row in rows), 3),
        "infer_ms_mean": round(mean(row.infer_ms for row in rows), 3),
        "postprocess_ms_mean": round(mean(row.postprocess_ms for row in rows), 3),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def series_x_values(rows: list[MetricsRow]) -> tuple[list[int], str]:
    frames = [row.frame for row in rows]
    if any(frame != 0 for frame in frames):
        return frames, "Frame"
    return list(range(1, len(rows) + 1)), "Sample"


def main() -> int:
    args = parse_args()
    config = plot_config(args)
    csv_path = resolve_csv_path(args, config)
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No usable rows found in {csv_path}")

    outdir = ensure_outdir(csv_path, args.outdir)
    x_values, x_label = series_x_values(rows)
    preprocess = [row.preprocess_ms for row in rows]
    inference = [row.infer_ms for row in rows]
    postprocess = [row.postprocess_ms for row in rows]

    plot_line(
        x_values,
        {
            "preprocess_ms": preprocess,
            "infer_ms": inference,
            "postprocess_ms": postprocess,
        },
        "Latency Breakdown Over Time",
        x_label,
        "Time (ms)",
        outdir / ("latency_breakdown_over_time.png"),
    )
    if config.regression_only:
        summary_path = write_regression_summary(rows, csv_path, outdir)
        print(f"Mode: REGRESSION")
        print(f"CSV: {csv_path}")
        print(f"Plots: {outdir}")
        print(f"Summary: {summary_path}")
        print(f"Frames: {len(rows)}")
        print(
            "Timing ms: "
            f"preprocess_mean={mean(preprocess):.3f}, "
            f"infer_mean={mean(inference):.3f}, "
            f"postprocess_mean={mean(postprocess):.3f}"
        )
        return 0

    latency = [row.latency_ms for row in rows]
    fps = [row.fps for row in rows]
    avg_conf_values = [row.avg_confidence for row in rows]
    min_conf_values = [row.min_confidence for row in rows]
    max_conf_values = [row.max_confidence for row in rows]

    plot_line(
        x_values,
        {"latency_ms": latency},
        "Total Latency Over Time",
        x_label,
        "Latency (ms)",
        outdir / "latency_over_time.png",
    )
    plot_line(
        x_values,
        {"fps": fps},
        "FPS Over Time",
        x_label,
        "FPS",
        outdir / "fps_over_time.png",
    )
    plot_hist(latency, "Latency Distribution", "Latency (ms)", outdir / "latency_histogram.png")

    summary_path = write_summary(rows, csv_path, outdir)

    print(f"CSV: {csv_path}")
    print(f"Plots: {outdir}")
    print(f"Summary: {summary_path}")
    print(f"Frames: {len(rows)}")
    print(f"Mean FPS: {mean(fps):.3f}")
    print(
        "Confidence: "
        f"avg_mean={mean(avg_conf_values):.6f}, "
        f"avg_p50={percentile(avg_conf_values, 50):.6f}, "
        f"avg_p95={percentile(avg_conf_values, 95):.6f}, "
        f"min_mean={mean(min_conf_values):.6f}, "
        f"max_mean={mean(max_conf_values):.6f}"
    )
    print(
        "Latency ms: "
        f"mean={mean(latency):.3f}, "
        f"p50={percentile(latency, 50):.3f}, "
        f"p95={percentile(latency, 95):.3f}, "
        f"p99={percentile(latency, 99):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
