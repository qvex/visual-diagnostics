from dataclasses import dataclass
from typing import Union
from pathlib import Path
import time

import torch
import cv2
import numpy as np
from ultralytics import YOLO


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(frozen=True)
class Success:
    value: any


@dataclass(frozen=True)
class Failure:
    error: str


Result = Union[Success, Failure]


def load_image(path: Path) -> Result:
    if not path.exists():
        return Failure(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        return Failure(f"Failed to read image: {path}")
    return Success(img)


def run_detection(model: YOLO, image: np.ndarray) -> tuple[list, float]:
    start = time.perf_counter()
    results = model(image, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return results, elapsed_ms


def save_annotated(results, output_path: Path) -> Result:
    annotated = results[0].plot()
    success = cv2.imwrite(str(output_path), annotated)
    if not success:
        return Failure(f"Failed to save: {output_path}")
    return Success(output_path)


def main(image_path: str, output_path: str = "results/output.png") -> None:
    image_result = load_image(Path(image_path))
    if isinstance(image_result, Failure):
        print(f"Error: {image_result.error}")
        return

    device = get_device()
    model = YOLO("yolov8n.pt")
    model.to(device)
    results, latency_ms = run_detection(model, image_result.value)

    print(f"Device: {device}")
    print(f"Inference time: {latency_ms:.2f} ms")
    print(f"Detections: {len(results[0].boxes)}")

    save_result = save_annotated(results, Path(output_path))
    if isinstance(save_result, Failure):
        print(f"Error: {save_result.error}")
    else:
        print(f"Saved to: {save_result.value}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_yolo.py <image_path> [output_path]")
        sys.exit(1)
    args = sys.argv[1:3] if len(sys.argv) > 2 else [sys.argv[1]]
    main(*args)
