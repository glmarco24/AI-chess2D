#!/usr/bin/env python3
"""Evaluate a chess-square checkpoint on the fixed generated test folder."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train import IMAGENET_MEAN, IMAGENET_STD, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/resnet18-v2/best.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("generated-test"))
    parser.add_argument("--output", type=Path, default=Path("generated-test/evaluation.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "resnet18":
        raise ValueError(
            f"{args.checkpoint} is not a ResNet-18 checkpoint; train a new model first"
        )
    classes = checkpoint["classes"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image_size = int(checkpoint.get("config", {}).get("image_size", 96))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    paths = sorted(args.dataset.glob("*/*.png"))
    if not paths:
        raise SystemExit(f"no test images found in {args.dataset}")
    images = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths])
    with torch.inference_mode():
        predicted_indices = model(images).argmax(1).tolist()

    by_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_set: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    errors: Counter[tuple[str, str]] = Counter()
    predictions: list[dict[str, object]] = []
    correct = 0
    for path, predicted_index in zip(paths, predicted_indices):
        truth = path.parent.name
        predicted = classes[predicted_index]
        is_correct = truth == predicted
        status = "OK" if is_correct else "WRONG"
        print(
            f"{path.relative_to(args.dataset)}: {status} "
            f"(expected={truth}, predicted={predicted})"
        )
        correct += is_correct
        by_class[truth][0] += is_correct
        by_class[truth][1] += 1
        by_set[path.stem][0] += is_correct
        by_set[path.stem][1] += 1
        if not is_correct:
            errors[truth, predicted] += 1
        predictions.append(
            {
                "path": str(path.relative_to(args.dataset)),
                "truth": truth,
                "predicted": predicted,
                "correct": is_correct,
            }
        )

    def summarized(groups: dict[str, list[int]]) -> dict[str, dict[str, object]]:
        return {
            name: {"correct": values[0], "total": values[1], "accuracy": values[0] / values[1]}
            for name, values in sorted(groups.items())
        }

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "correct": correct,
        "total": len(paths),
        "accuracy": correct / len(paths),
        "by_class": summarized(by_class),
        "by_set": summarized(by_set),
        "errors": [
            {"truth": truth, "predicted": predicted, "count": count}
            for (truth, predicted), count in errors.most_common()
        ],
        "predictions": predictions,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Accuracy: {correct}/{len(paths)} = {correct / len(paths):.2%}")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
