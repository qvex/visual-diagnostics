from dataclasses import dataclass
from typing import Union, Iterator
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Success:
    value: any


@dataclass(frozen=True)
class Failure:
    error: str


Result = Union[Success, Failure]


@dataclass
class MVTecSample:
    image: np.ndarray
    mask: np.ndarray | None
    label: str
    image_path: Path


def get_mvtec_categories(data_dir: Path) -> list[str]:
    return [d.name for d in data_dir.iterdir() if d.is_dir()]


def load_image(path: Path) -> Result:
    if not path.exists():
        return Failure(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        return Failure(f"Failed to read: {path}")
    return Success(img)


def load_mask(mask_path: Path) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return mask


def iter_test_samples(
    data_dir: Path,
    category: str
) -> Iterator[Result]:
    category_dir = data_dir / category / "test"
    if not category_dir.exists():
        yield Failure(f"Category not found: {category}")
        return

    for defect_type in category_dir.iterdir():
        if not defect_type.is_dir():
            continue
        label = defect_type.name
        gt_dir = data_dir / category / "ground_truth" / label

        for img_path in sorted(defect_type.glob("*.png")):
            img_result = load_image(img_path)
            if isinstance(img_result, Failure):
                yield img_result
                continue

            mask_path = gt_dir / img_path.name.replace(".png", "_mask.png")
            mask = load_mask(mask_path)

            yield Success(MVTecSample(
                image=img_result.value,
                mask=mask,
                label=label,
                image_path=img_path
            ))


def count_samples(data_dir: Path, category: str) -> dict[str, int]:
    counts = {}
    category_dir = data_dir / category / "test"
    for defect_type in category_dir.iterdir():
        if defect_type.is_dir():
            counts[defect_type.name] = len(list(defect_type.glob("*.png")))
    return counts


if __name__ == "__main__":
    data_dir = Path("data/mvtec_ad")
    category = "transistor"

    print(f"Categories: {get_mvtec_categories(data_dir)}")
    print(f"\n{category} sample counts:")
    for defect, count in count_samples(data_dir, category).items():
        print(f"  {defect}: {count}")

    print(f"\nLoading first 3 samples from {category}:")
    for i, result in enumerate(iter_test_samples(data_dir, category)):
        if i >= 3:
            break
        if isinstance(result, Success):
            sample = result.value
            mask_info = f"mask {sample.mask.shape}" if sample.mask is not None else "no mask"
            print(f"  {sample.label}: {sample.image.shape}, {mask_info}")
        else:
            print(f"  Error: {result.error}")
