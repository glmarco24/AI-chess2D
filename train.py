#!/usr/bin/env python3
"""Fine-tune ResNet-18 to classify one chessboard square.

STEPS 1-8 COMPLETE
The pipeline validates the dataset, builds data loaders, fine-tunes with
validation, saves the best checkpoint, then evaluates it once on the untouched
test split.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainingConfig:
    """All values that should describe and reproduce one training run."""

    dataset_dir: Path = Path("dataset")
    output_dir: Path = Path("runs/resnet18-v2")
    image_size: int = 64
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-3
    label_smoothing: float = 0.05
    num_workers: int = 0
    seed: int = 20260620
    pretrained: bool = True


@dataclass(frozen=True)
class EpochMetrics:
    """Average loss and classification accuracy from one complete data pass."""

    loss: float
    accuracy: float


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    """Build ResNet-18 and replace its ImageNet classification head."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Load dataset.json so training uses the exact classes and image dimensions
# recorded when the augmented dataset was generated.
def read_dataset_metadata(dataset_dir: Path) -> dict[str, object]:
    """Read the contract produced by build_dataset.py."""
    metadata_path = dataset_dir / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run build_dataset.py before training."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


# Check that all three split directories and their class directories exist and
# that the dataset has square images. Return class
# names in their canonical order because that order defines numeric labels.
def validate_dataset(config: TrainingConfig, metadata: dict[str, object]) -> list[str]:
    """Fail early when dataset assumptions do not match the training setup."""
    classes = metadata.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("dataset.json has no valid 'classes' list")

    dataset_size = metadata.get("image_size")
    if (
        not isinstance(dataset_size, list)
        or len(dataset_size) != 2
        or dataset_size[0] != dataset_size[1]
    ):
        raise ValueError(
            f"dataset.json has invalid square image dimensions: {dataset_size}"
        )

    for split in ("train", "validation", "test"):
        split_dir = config.dataset_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset split: {split_dir}")
        missing = [name for name in classes if not (split_dir / name).is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing classes in {split}: {missing}")

    return [str(name) for name in classes]


# Initialize Python and PyTorch random-number generators with the same seed so
# repeated runs begin with the same shuffle order and future model weights.
def seed_everything(seed: int) -> None:
    """Seed the random-number generators currently used by this script."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Select the fastest supported computation backend: NVIDIA CUDA first, Apple
# Metal second, and the CPU as the universally available fallback.
def choose_device() -> torch.device:
    """Prefer a GPU when PyTorch can use one, otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Convert directory-based image splits into batched PyTorch data loaders. Each
# image becomes a normalized 3×96×96 tensor; training is shuffled while
# validation and test ordering stays fixed.
def build_dataloaders(
    config: TrainingConfig, expected_classes: list[str]
) -> dict[str, DataLoader]:
    """Create loaders for the existing source-separated dataset splits."""
    # Match the normalization used to pretrain ResNet-18 on ImageNet.
    image_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    image_folder_datasets = {
        split: datasets.ImageFolder(config.dataset_dir / split, transform=image_transform)
        for split in ("train", "validation", "test")
    }
    for split, dataset in image_folder_datasets.items():
        if dataset.classes != expected_classes:
            raise ValueError(
                f"{split} class order is {dataset.classes}, expected {expected_classes}"
            )

    shuffle_generator = torch.Generator().manual_seed(config.seed)
    return {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
            generator=shuffle_generator if split == "train" else None,
        )
        for split, dataset in image_folder_datasets.items()
    }


# Count only trainable tensors to show the actual capacity being optimized;
# BatchNorm's running averages are state, but they are not trainable parameters.
def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of scalar model parameters updated by the optimizer."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


# Verify architecture wiring before a long run. A wrong channel count or
# flattening size should fail here, before any optimizer state is created.
def check_model_output(
    model: nn.Module, images: torch.Tensor, num_classes: int, device: torch.device
) -> tuple[int, ...]:
    """Run one inference-only batch and validate the logits tensor shape."""
    model.eval()
    with torch.inference_mode():
        logits = model(images.to(device))
    expected_shape = (images.shape[0], num_classes)
    if tuple(logits.shape) != expected_shape:
        raise ValueError(f"Model returned {tuple(logits.shape)}, expected {expected_shape}")
    return tuple(logits.shape)


# Perform one full learning pass over the training split. Every batch follows
# the essential training sequence: clear old gradients, predict, measure loss,
# backpropagate derivatives, and update weights.
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    """Update model weights using every training batch exactly once."""
    model.train()  # Enables training behavior in BatchNorm and Dropout.
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Multiplying batch-average loss by batch size lets us calculate the
        # correct dataset-wide average even when the last batch is smaller.
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )


# Measure a split without gradients or weight updates. The confusion matrix
# stores actual classes in rows and predicted classes in columns, making later
# per-class accuracy calculation possible without another model pass.
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[EpochMetrics, torch.Tensor]:
    """Calculate loss, accuracy, and confusion matrix for validation or test."""
    model.eval()  # Uses fixed BatchNorm statistics and disables Dropout.
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    # inference_mode saves memory because evaluation never needs derivatives.
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_examples += batch_size

            # Encoding each (actual, prediction) pair as one integer lets
            # bincount update the whole confusion matrix without a Python loop.
            encoded = labels.cpu() * num_classes + predictions.cpu()
            confusion += torch.bincount(
                encoded, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    return (
        EpochMetrics(
            loss=total_loss / total_examples,
            accuracy=total_correct / total_examples,
        ),
        confusion,
    )


# Store the best weights together with the label order needed to interpret the
# output neurons. Saving state_dict rather than the model object keeps the file
# portable and makes the architecture remain explicit in source code.
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    validation_metrics: EpochMetrics,
    classes: list[str],
    config: TrainingConfig,
) -> None:
    """Write a resumable best-model checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_config = {
        **asdict(config),
        "dataset_dir": str(config.dataset_dir),
        "output_dir": str(config.output_dir),
    }
    torch.save(
        {
            "architecture": "resnet18",
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "validation": asdict(validation_metrics),
            "classes": classes,
            "config": serializable_config,
        },
        path,
    )


# Reload only the learned tensors from our own checkpoint. Keeping this as a
# separate function makes it explicit that final test results use the epoch
# selected by validation accuracy, not simply the last training epoch.
def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, object]:
    """Load checkpoint data and copy its best weights into the model."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


# Convert the confusion-matrix diagonal into readable accuracy for each class.
# This reveals failures hidden by a strong overall accuracy number.
def per_class_accuracy(
    confusion: torch.Tensor, classes: list[str]
) -> dict[str, float]:
    """Return correct/total accuracy separately for every class."""
    totals = confusion.sum(dim=1)
    return {
        class_name: (
            confusion[index, index].item() / totals[index].item()
            if totals[index].item()
            else 0.0
        )
        for index, class_name in enumerate(classes)
    }


# Persist human-readable training history and final test measurements as JSON,
# while the binary checkpoint remains dedicated to PyTorch tensors.
def write_results(
    path: Path,
    history: list[dict[str, float]],
    test_metrics: EpochMetrics,
    confusion: torch.Tensor,
    class_accuracy: dict[str, float],
) -> None:
    """Save training and test metrics for later comparison between runs."""
    result = {
        "history": history,
        "test": asdict(test_metrics),
        "per_class_accuracy": class_accuracy,
        "confusion_matrix": confusion.tolist(),
    }
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


# Read command-line path overrides while keeping the default dataset and run
# directories convenient for normal use from the project root. Hyperparameter
# flags make experiments possible without editing source code.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("runs/resnet18-v2"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--image-size", type=int, default=64,
        help="resize input images during loading (source PNG files are unchanged)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="initialize ResNet-18 randomly instead of using ImageNet weights",
    )
    parser.set_defaults(pretrained=True)
    return parser.parse_args()


# Assemble the full pipeline in a deliberate order: validate inputs, construct
# data and model, sanity-check shapes, train while selecting on validation, and
# finally evaluate the selected model once on the untouched test set.
def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32 or args.learning_rate <= 0:
        raise SystemExit("epochs, batch size, image size (>= 32), and learning rate must be positive")
    if args.weight_decay < 0 or args.num_workers < 0:
        raise SystemExit("weight decay and number of workers cannot be negative")
    if not 0 <= args.label_smoothing < 1:
        raise SystemExit("label smoothing must be in [0, 1)")
    if not 0 <= args.min_learning_rate <= args.learning_rate:
        raise SystemExit("minimum learning rate must be between zero and learning rate")

    config = TrainingConfig(
        dataset_dir=args.dataset,
        output_dir=args.output,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained=args.pretrained,
    )
    seed_everything(config.seed)
    metadata = read_dataset_metadata(config.dataset_dir)
    classes = validate_dataset(config, metadata)
    device = choose_device()
    loaders = build_dataloaders(config, classes)

    # Move the model once; every future batch must be moved to the same device.
    model = build_model(num_classes=len(classes), pretrained=config.pretrained).to(device)

    # CrossEntropyLoss combines log-softmax with negative log likelihood. AdamW
    # provides adaptive updates and decoupled weight decay regularization.
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_learning_rate,
    )

    # Use validation data for the shape check so we do not consume or alter the
    # deterministic first shuffle of the training loader.
    sample_images, sample_labels = next(iter(loaders["validation"]))
    output_shape = check_model_output(model, sample_images, len(classes), device)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "best.pt"
    results_path = config.output_dir / "metrics.json"
    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_loss = float("inf")

    print("Training configuration")
    print(f"  device: {device}")
    print(f"  classes: {len(classes)}")
    print(f"  train/validation/test: "
          f"{len(loaders['train'].dataset)}/"
          f"{len(loaders['validation'].dataset)}/"
          f"{len(loaders['test'].dataset)}")
    print(f"  resized input: {config.image_size}x{config.image_size}")
    print(f"  sample input: {tuple(sample_images.shape)}")
    print(f"  sample labels: {tuple(sample_labels.shape)}")
    print(f"  model output: {output_shape}")
    print(f"  trainable parameters: {count_trainable_parameters(model):,}")
    print(f"  label smoothing: {config.label_smoothing}")
    print(f"  learning rate: {config.learning_rate:g} -> {config.min_learning_rate:g} (cosine)")

    for epoch in range(1, config.epochs + 1):
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        validation_metrics, _ = evaluate(
            model, loaders["validation"], criterion, device, len(classes)
        )
        history.append(
            {
                "epoch": float(epoch),
                "learning_rate": epoch_learning_rate,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "validation_loss": validation_metrics.loss,
                "validation_accuracy": validation_metrics.accuracy,
            }
        )

        print(
            f"Epoch {epoch:02d}/{config.epochs}: "
            f"lr={epoch_learning_rate:.2e}; "
            f"train loss={train_metrics.loss:.4f}, "
            f"accuracy={train_metrics.accuracy:.2%}; "
            f"validation loss={validation_metrics.loss:.4f}, "
            f"accuracy={validation_metrics.accuracy:.2%}"
        )

        # Validation accuracy is the primary selection criterion. Lower loss
        # breaks exact accuracy ties because it reflects prediction confidence.
        is_better = validation_metrics.accuracy > best_accuracy or (
            validation_metrics.accuracy == best_accuracy
            and validation_metrics.loss < best_loss
        )
        if is_better:
            best_accuracy = validation_metrics.accuracy
            best_loss = validation_metrics.loss
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                validation_metrics,
                classes,
                config,
            )
            print(f"  saved new best checkpoint: {checkpoint_path}")

        scheduler.step()

    # Testing happens only after model selection. Looking at test performance
    # during training would indirectly turn the test set into validation data.
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    if checkpoint["classes"] != classes:
        raise ValueError("checkpoint class order does not match this dataset")
    test_metrics, confusion = evaluate(
        model, loaders["test"], criterion, device, len(classes)
    )
    class_accuracy = per_class_accuracy(confusion, classes)
    write_results(results_path, history, test_metrics, confusion, class_accuracy)

    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Test loss: {test_metrics.loss:.4f}")
    print(f"Test accuracy: {test_metrics.accuracy:.2%}")
    for class_name, accuracy in class_accuracy.items():
        print(f"  {class_name:14s} {accuracy:.2%}")
    print(f"Metrics written to {results_path}")


if __name__ == "__main__":
    main()
