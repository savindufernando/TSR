# Traffic Sign Recognition (GTSRB) with CNN + ViT
A high-performance hybrid model for road sign classification.

## Usage

This project trains a **hybrid MobileNetV2 (CNN) + Transformer encoder (ViT-like)** classifier for **German Traffic Sign Recognition Benchmark (GTSRB)**, and supports exporting an optimized **TensorFlow Lite** model for edge deployment.

## 1) Dataset (KaggleHub)

This code is written to work with the Kaggle dataset:

```python
import kagglehub

path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
print("Path to dataset files:", path)
```

The dataset folder typically contains a `Train/` directory (class subfolders `0..42`) and either:
- `Test/` as class-subfolders, or
- a `Test.csv` file listing image paths + labels.

Summarize class counts quickly:

```bash
python scripts/summarize_dataset.py --data "$(Get-Content data/dataset_path.txt)"
```

## 2) Install

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

TensorFlow currently does not publish Windows wheels for Python `3.13+`. Use Python `3.10`, `3.11`, or `3.12` (recommended: `3.11`) to train/export this model.

## 3) Train

Download the dataset (requires KaggleHub auth) and train:

```bash
python scripts/download_dataset.py --out data
python scripts/train.py --data "$(Get-Content data/dataset_path.txt)" --img-size 224 --batch-size 64 --epochs 50
```

If you’re on bash/zsh, use `--data "$(cat data/dataset_path.txt)"`.

Use `--seed <int>` to set Python/NumPy/TensorFlow seeds for reproducible training runs (default: `1337`).

Outputs are written to `outputs/` by default (SavedModel + logs + metrics).

Add `--cache` to keep train/val/test datasets in memory; this speeds up small experiments but requires enough RAM.

Tune over- or under-confidence with `--label-smoothing <0..1>` (e.g., `0.1`); default is no smoothing.

Mitigate class imbalance with `--use-class-weights`, which computes weights from the `Train/` folder and passes them to `model.fit`.

Skip augmentation with `--no-augment` if you only want normalization applied to images.

Speed up training on Ampere+ GPUs/CPUs by enabling automatic mixed precision: `--mixed-precision`.

## 4) Evaluate

```bash
python scripts/evaluate.py --data "$(Get-Content data/dataset_path.txt)" --model outputs/saved_model
```

This prints accuracy and per-class precision/recall/F1.

## 5) Export to TensorFlow Lite

Default dynamic-range optimization:

```bash
python scripts/export_tflite.py --saved-model outputs/saved_model --out outputs/model.tflite
```

Optional full integer quantization (needs a representative dataset):

```bash
python scripts/export_tflite.py --saved-model outputs/saved_model --out outputs/model_int8.tflite --int8 --data "$(Get-Content data/dataset_path.txt)"
```

## 6) Quick inference (SavedModel)

Run prediction for a single image with a SavedModel or `.keras` file:

```bash
python scripts/predict_image.py --model outputs/saved_model --image path/to/image.png --img-size 224 --class-names data/class_names.txt
```

`--class-names` is optional; if omitted, class indices are used.

## 7) Quick inference (TFLite)

Run an exported `.tflite` model:

```bash
python scripts/predict_tflite.py --model outputs/model.tflite --image path/to/image.png --img-size 224 --class-names data/class_names.txt
```

## Notes

- The “ViT” part is implemented as a **Transformer encoder over CNN feature-map tokens** (a common hybrid CNN+Transformer design) so you get both local features and global context while staying lightweight.
- For real-time use, pair this classifier with a detector/ROI cropper (or use a pipeline that detects signs first, then classifies).
