from dataclasses import dataclass
from pathlib import Path
import time

import torch
import numpy as np
from anomalib.models import Patchcore
from anomalib.data import MVTecAD
from torchvision.transforms import v2
import lightning.pytorch as pl

from load_mvtec import Success, Failure


@dataclass
class PatchCoreResult:
    label: str
    image_path: str
    anomaly_score: float
    inference_ms: float
    has_anomaly: bool


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_patchcore(data_dir: Path, category: str):
    transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    datamodule = MVTecAD(
        root=data_dir,
        category=category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,
        augmentations=transform,
    )
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        default_root_dir="results/patchcore",
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model=model, datamodule=datamodule)
    return trainer, model, datamodule


def run_patchcore_benchmark(data_dir: Path, category: str) -> list[PatchCoreResult]:
    print(f"Training PatchCore on {category}...")
    trainer, model, datamodule = train_patchcore(data_dir, category)

    print(f"Running inference on test set...")
    results = []

    test_predictions = trainer.predict(model=model, datamodule=datamodule)

    for batch in test_predictions:
        batch_size = len(batch.image_path)
        for i in range(batch_size):
            img_path = batch.image_path[i]
            label = Path(img_path).parent.name
            score = batch.pred_score[i].item()

            results.append(PatchCoreResult(
                label=label,
                image_path=str(img_path),
                anomaly_score=score,
                inference_ms=0,
                has_anomaly=(label != "good")
            ))

    return results


def calculate_auroc(results: list[PatchCoreResult]) -> float:
    from sklearn.metrics import roc_auc_score
    y_true = [1 if r.has_anomaly else 0 for r in results]
    y_scores = [r.anomaly_score for r in results]
    return roc_auc_score(y_true, y_scores)


def print_summary(results: list[PatchCoreResult]) -> None:
    if not results:
        print("No results")
        return

    auroc = calculate_auroc(results)

    by_label = {}
    for r in results:
        if r.label not in by_label:
            by_label[r.label] = []
        by_label[r.label].append(r)

    print(f"\n{'='*50}")
    print(f"PatchCore Benchmark Results")
    print(f"{'='*50}")
    print(f"Total samples: {len(results)}")
    print(f"Device: {get_device()}")
    print(f"\nImage-level AUROC: {auroc:.4f}")

    print(f"\nBy defect type (avg anomaly score):")
    for label, items in sorted(by_label.items()):
        avg_score = sum(r.anomaly_score for r in items) / len(items)
        print(f"  {label}: {len(items)} samples, avg score: {avg_score:.4f}")


def save_results_csv(results: list[PatchCoreResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("label,has_anomaly,anomaly_score,image_path\n")
        for r in results:
            f.write(f"{r.label},{r.has_anomaly},{r.anomaly_score:.6f},{r.image_path}\n")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    data_dir = Path("data/mvtec_ad")
    category = "transistor"

    print(f"Running PatchCore benchmark on {category}...")
    results = run_patchcore_benchmark(data_dir, category)

    print_summary(results)
    save_results_csv(results, Path("results/patchcore_benchmark.csv"))
