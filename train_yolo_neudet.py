from pathlib import Path
import time

import torch
from ultralytics import YOLO


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_yolo(data_yaml: Path, epochs: int = 50) -> YOLO:
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        project="results/yolo_neudet",
        name="train",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    return model


def evaluate_yolo(model: YOLO, data_yaml: Path) -> dict:
    results = model.val(
        data=str(data_yaml),
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=True,
    )
    return {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
    }


def measure_latency(model: YOLO, img_path: Path, n_runs: int = 100) -> float:
    for _ in range(10):
        model(str(img_path), verbose=False)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model(str(img_path), verbose=False)
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times)


def print_results(metrics: dict, latency_ms: float) -> None:
    print(f"\n{'='*50}")
    print("YOLOv8 Fine-tuned on NEU-DET Results")
    print(f"{'='*50}")
    print(f"Device: {get_device()}")
    print(f"\nDetection Metrics:")
    print(f"  mAP@50:    {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"\nLatency:")
    print(f"  Avg: {latency_ms:.2f} ms")
    print(f"  FPS: {1000/latency_ms:.1f}")


if __name__ == "__main__":
    data_yaml = Path("data/neu-det-yolo/neu-det.yaml")

    print("Training YOLOv8 on NEU-DET...")
    model = train_yolo(data_yaml, epochs=50)

    print("\nEvaluating on validation set...")
    metrics = evaluate_yolo(model, data_yaml)

    print("\nMeasuring latency...")
    sample_img = Path("data/neu-det-yolo/images/validation/crazing_241.jpg")
    latency_ms = measure_latency(model, sample_img)

    print_results(metrics, latency_ms)
