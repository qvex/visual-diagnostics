from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score


CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


class NEUDataset(Dataset):
    def __init__(self, root: Path, split: str, transform=None):
        self.transform = transform
        self.samples = []
        img_dir = root / "images" / split
        for img_path in img_dir.glob("*.jpg"):
            class_name = img_path.stem.rsplit("_", 1)[0]
            if class_name in CLASSES:
                self.samples.append((img_path, CLASSES.index(class_name)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_model(num_classes: int):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


def measure_latency(model, device, n_runs: int = 100) -> float:
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(10):
        model(dummy)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model(dummy)
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)


def print_results(acc: float, f1: float, latency_ms: float) -> None:
    print(f"\nEfficientNet-B0 NEU-DET Results [{get_device()}]")
    print(f"  Accuracy: {acc:.4f}  |  F1 (macro): {f1:.4f}")
    print(f"  Latency: {latency_ms:.2f} ms ({1000/latency_ms:.1f} FPS)")


if __name__ == "__main__":
    device = get_device()
    data_dir = Path("data/neu-det-yolo")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = NEUDataset(data_dir, "train", transform)
    val_ds = NEUDataset(data_dir, "validation", transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"Training EfficientNet-B0 on NEU-DET ({len(train_ds)} train, {len(val_ds)} val)...")

    model = create_model(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        acc, f1 = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch+1}/10  |  Loss: {loss:.4f}  |  Val Acc: {acc:.4f}")

    print("\nFinal evaluation...")
    acc, f1 = evaluate(model, val_loader, device)
    latency_ms = measure_latency(model, device)
    print_results(acc, f1, latency_ms)

    torch.save(model.state_dict(), "results/efficientnet_neudet.pt")
    print(f"  Model saved to results/efficientnet_neudet.pt")
