from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Make imports work when running: python analysis/robustness.py
ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.alexnet import AlexNet  # noqa: E402
from models.lenet import LeNet  # noqa: E402
from models.resnet import ResNet18  # noqa: E402


def get_device() -> torch.device:
    print("Using CPU for robustness analysis")
    return torch.device("cpu")


def get_test_loader(batch_size: int = 64) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    test_dataset = datasets.CIFAR10(
        root=str(PROJECT_ROOT / "data"),
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def build_model(model_name: str, device: torch.device) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "lenet":
        model = LeNet(num_classes=10).to(device)
        ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / "lenet_best.pth"
    elif model_name == "alexnet":
        model = AlexNet(num_classes=10).to(device)
        ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / "alexnet_best.pth"
    elif model_name == "resnet":
        model = ResNet18(num_classes=10).to(device)
        ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / "resnet_best.pth"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run training for {model_name} first."
        )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, noise_std: float = 0.0) -> float:
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if noise_std > 0:
            noise = torch.randn_like(images) * noise_std
            images = images + noise

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lenet", "alexnet", "resnet"],
        help="Which model to evaluate",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.15,
        help="Gaussian noise standard deviation",
    )
    args = parser.parse_args()

    device = get_device()
    loader = get_test_loader(batch_size=64)
    model = build_model(args.model, device)

    clean_acc = evaluate(model, loader, device, noise_std=0.0)
    noisy_acc = evaluate(model, loader, device, noise_std=args.noise_std)

    results = {
        "model": args.model,
        "noise_std": args.noise_std,
        "clean_acc": clean_acc,
        "noisy_acc": noisy_acc,
        "drop": clean_acc - noisy_acc,
    }

    out_dir = PROJECT_ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_robustness.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Clean Acc: {clean_acc:.4f}")
    print(f"Noisy Acc: {noisy_acc:.4f}")
    print(f"Drop: {clean_acc - noisy_acc:.4f}")
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()