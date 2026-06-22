# Chess piece set preparation

`prepare_sets.py` detects the 8×8 board in each PNG, removes the screenshot
margin, slices all 64 squares, and selects one example of every piece plus a
blank square. It expects the normal starting position with Black at the top.

```bash
python -m pip install -r requirements.txt
python prepare_sets.py
```

Output is written to `sets-prepared/<screenshot>/`:

- `board.png` — the cropped board only
- `squares/a1.png` through `squares/h8.png` — all 64 squares
- `pieces/*.png` — the 12 pieces and one blank square
- `sets-prepared/manifest.json` — crop coordinates and extraction details

Pass a single PNG or a different directory as the first argument. Use `-o` to
change the destination:

```bash
python prepare_sets.py path/to/screenshots -o path/to/output
```

## Build the training dataset

The augmentation script creates balanced `train`, `validation`, and `test`
class directories. Splits are made by original piece set, not by generated
image, which prevents augmented versions of the same visual style leaking into
validation or test data.

```bash
python build_dataset.py
```

The default produces 28,080 labeled 96×96 PNG files in `dataset/`. Every piece
is composited onto colors sampled from all source boards, translated by up to
6 pixels, and mildly blurred or sharpened. Coherent synthetic-set styles also
vary piece scale, aspect ratio, stroke thickness, edge rendering, contrast,
opacity, tint, outline, shadow, and subtle background texture. All 12 pieces in
one source/variant share the same style parameters. `dataset/manifest.csv`
records every image and its augmentation parameters; `dataset/dataset.json`
records classes, split assignments, augmentation ranges, and counts.

Generation is deterministic. To change its size or replace an existing build:

```bash
python build_dataset.py --per-source 80 --size 96 --max-shift 6 --force
```

## Train the classifier

Install PyTorch, then run the complete training/validation/test pipeline:

```bash
python train.py
```

Training fine-tunes an ImageNet-pretrained ResNet-18. The first run downloads
the pretrained weights. The best validation checkpoint is saved as
`runs/resnet18-v2/best.pt`. Training uses 0.05 label smoothing and cosine
learning-rate decay by default. Images are resized to 64×64 during loading,
while every training and validation image is used. All ResNet-18 parameters
are fine-tuned with a batch size of 128. The defaults run for 10 epochs. Training
history, test accuracy, per-class accuracy, and the confusion matrix are saved
as `runs/resnet18-v2/metrics.json`. Common hyperparameters can be overridden
without editing the script:

```bash
python train.py --epochs 30 --batch-size 128 --learning-rate 0.0001 \
  --min-learning-rate 0.000001 --label-smoothing 0.05 --weight-decay 0.001
```

Pass `--no-pretrained` only when intentionally training ResNet-18 from random
initialization.
