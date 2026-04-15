from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Make imports work when running: python src/train.py
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.lenet import LeNet
from models.alexnet import AlexNet
from models.resnet import ResNet18

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 0,
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=test_transform,
    )

    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += batch_size
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += batch_size
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def save_history_plot(history: dict, output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    set_seed(42)

    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "results"
    checkpoints_dir = results_dir / "checkpoints"
    metrics_dir = results_dir / "metrics"
    figures_dir = PROJECT_ROOT / "figures"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=128,
        val_ratio=0.1,
        num_workers=0,  # best default on macOS
    )

    model_name = "resnet"  # 在这里选择模型

    if model_name == "lenet":
        model = LeNet(num_classes=10).to(device)
    elif model_name == "alexnet":
        model = AlexNet(num_classes=10).to(device)
    elif model_name == "resnet":
        model = ResNet18(num_classes=10).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    best_val_acc = 0.0
    best_epoch = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_ckpt_path = checkpoints_dir / f"{model_name}_best.pth"
    metrics_path = metrics_dir / f"{model_name}_metrics.json"
    plot_path = figures_dir / f"{model_name}_training_curves.png"

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                best_ckpt_path,
            )

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    metrics = {
        "model": model_name,
        "dataset": "CIFAR-10",
        "num_epochs": num_epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": history,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    save_history_plot(history, plot_path)

    print(f"Saved checkpoint to: {best_ckpt_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved training curves to: {plot_path}")


if __name__ == "__main__":
    main()