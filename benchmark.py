from dataclasses import dataclass
from typing import Union
from pathlib import Path
import time

import torch
import cv2
import numpy as np
from ultralytics import YOLO

from load_mvtec import iter_test_samples, count_samples, Success, Failure


@dataclass
class BenchmarkResult:
    category: str
    label: str
    image_path: str
    inference_ms: float
    num_detections: int
    has_anomaly: bool


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_yolo_benchmark(
    data_dir: Path,
    category: str,
    model_name: str = "yolov8n.pt"
) -> list[BenchmarkResult]:
    device = get_device()
    model = YOLO(model_name)
    model.to(device)

    results = []
    for i in range(3):
        _ = model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    for sample_result in iter_test_samples(data_dir, category):
        if isinstance(sample_result, Failure):
            print(f"Error: {sample_result.error}")
            continue

        sample = sample_result.value
        start = time.perf_counter()
        detections = model(sample.image, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

        num_boxes = len(detections[0].boxes)
        has_anomaly = sample.label != "good"

        results.append(BenchmarkResult(
            category=category,
            label=sample.label,
            image_path=str(sample.image_path),
            inference_ms=elapsed_ms,
            num_detections=num_boxes,
            has_anomaly=has_anomaly
        ))

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    if not results:
        print("No results")
        return

    latencies = [r.inference_ms for r in results]
    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)

    by_label = {}
    for r in results:
        if r.label not in by_label:
            by_label[r.label] = []
        by_label[r.label].append(r)

    print(f"\n{'='*50}")
    print(f"YOLO Benchmark Results")
    print(f"{'='*50}")
    print(f"Total samples: {len(results)}")
    print(f"Device: {get_device()}")
    print(f"\nLatency:")
    print(f"  Avg: {avg_ms:.2f} ms")
    print(f"  Min: {min_ms:.2f} ms")
    print(f"  Max: {max_ms:.2f} ms")
    print(f"  FPS: {1000/avg_ms:.1f}")

    print(f"\nBy defect type:")
    for label, items in sorted(by_label.items()):
        avg_det = sum(r.num_detections for r in items) / len(items)
        print(f"  {label}: {len(items)} samples, avg detections: {avg_det:.1f}")


def save_results_csv(results: list[BenchmarkResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("category,label,has_anomaly,inference_ms,num_detections,image_path\n")
        for r in results:
            f.write(f"{r.category},{r.label},{r.has_anomaly},{r.inference_ms:.2f},")
            f.write(f"{r.num_detections},{r.image_path}\n")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    data_dir = Path("data/mvtec_ad")
    category = "transistor"

    print(f"Sample counts for {category}:")
    for defect, count in count_samples(data_dir, category).items():
        print(f"  {defect}: {count}")

    print(f"\nRunning YOLO benchmark on {category}...")
    results = run_yolo_benchmark(data_dir, category)

    print_summary(results)
    save_results_csv(results, Path("results/yolo_benchmark.csv"))
