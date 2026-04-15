from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Make imports work when running: python analysis/tsne.py
ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.alexnet import AlexNet  # noqa: E402
from models.lenet import LeNet  # noqa: E402
from models.resnet import ResNet18  # noqa: E402


def get_device() -> torch.device:
    # Use CPU for stability in sklearn PCA analysis.
    print("Using CPU for representation analysis")
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


def build_model_and_feature_layer(
    model_name: str,
    device: torch.device,
):
    model_name = model_name.lower()

    if model_name == "lenet":
        model = LeNet(num_classes=10).to(device)
        ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / "lenet_best.pth"
        feature_layer = model.fc2

    elif model_name == "alexnet":
        model = AlexNet(num_classes=10).to(device)
        ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / "alexnet_best.pth"
        feature_layer = model.classifier[4]

    elif model_name == "resnet":
        model = ResNet18(num_classes=10).to(device)
        ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / "resnet_best.pth"
        feature_layer = model.avgpool

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
    return model, feature_layer, ckpt_path


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    feature_layer: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    collected = 0

    def hook_fn(module, inp, out):
        feat = out
        if feat.dim() == 4:
            feat = torch.flatten(feat, 1)
        features.append(feat.detach().cpu())

    handle = feature_layer.register_forward_hook(hook_fn)

    for images, target in loader:
        images = images.to(device)
        target = target.cpu()

        _ = model(images)

        batch_feat = features[-1]
        batch_size = batch_feat.shape[0]

        if collected + batch_size > max_samples:
            keep = max_samples - collected
            features[-1] = batch_feat[:keep]
            labels.append(target[:keep])
            collected += keep
            break
        else:
            labels.append(target)
            collected += batch_size

        if collected >= max_samples:
            break

    handle.remove()

    feats = torch.cat(features, dim=0).numpy().astype(np.float32)
    labs = torch.cat(labels, dim=0).numpy()

    feats = feats[:max_samples]
    labs = labs[:max_samples]
    return feats, labs


def plot_pca(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    model_name: str,
) -> None:
    print("Running PCA...")

    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(features)

    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance ratio (2D): {explained:.4f}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=10, cmap="tab10")
    plt.colorbar(scatter, ticks=range(10))
    plt.title(f"PCA of {model_name.upper()} Features on CIFAR-10")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved PCA plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="resnet",
        choices=["lenet", "alexnet", "resnet"],
        help="Which model to analyze",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Number of samples to use for PCA",
    )
    args = parser.parse_args()

    device = get_device()
    loader = get_test_loader(batch_size=64)
    model, feature_layer, _ = build_model_and_feature_layer(args.model, device)

    features, labels = extract_features(
        model=model,
        feature_layer=feature_layer,
        loader=loader,
        device=device,
        max_samples=args.max_samples,
    )

    fig_dir = PROJECT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / f"{args.model}_pca.png"

    plot_pca(features, labels, output_path, args.model)


if __name__ == "__main__":
    main()