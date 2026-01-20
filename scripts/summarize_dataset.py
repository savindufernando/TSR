from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple


def _find_dir(root: Path, candidates: Tuple[str, str]) -> Path | None:
    for name in candidates:
        candidate = root / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _count_class_dirs(split_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        counts[class_dir.name] = len([p for p in class_dir.iterdir() if p.is_file()])
    return counts


def _count_test_csv(csv_path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = {name.lower(): name for name in (reader.fieldnames or [])}
        class_key = (
            fieldnames.get("classid")
            or fieldnames.get("class_id")
            or fieldnames.get("label")
            or fieldnames.get("class")
        )
        if not class_key:
            return counts
        for row in reader:
            cls = row[class_key]
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def _print_counts(title: str, counts: Dict[str, int]):
    if not counts:
        print(f"{title}: none")
        return
    total = sum(counts.values())
    print(f"{title}: {total} images across {len(counts)} classes")
    try:
        sorted_items = sorted(counts.items(), key=lambda kv: int(kv[0]))
    except ValueError:
        sorted_items = sorted(counts.items(), key=lambda kv: kv[0])
    for name, count in sorted_items:
        print(f"  {name}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize GTSRB-style dataset directory.")
    parser.add_argument("--data", type=str, required=True, help="Dataset root containing Train/")
    args = parser.parse_args()

    dataset_root = Path(args.data)
    train_dir = _find_dir(dataset_root, ("Train", "train"))
    if train_dir is None:
        raise SystemExit(f"Could not find Train/ under {dataset_root}")

    test_dir = _find_dir(dataset_root, ("Test", "test"))
    train_counts = _count_class_dirs(train_dir)
    test_counts = _count_class_dirs(test_dir) if test_dir else {}

    # Fall back to Test.csv when no Test/ directory is present.
    if not test_counts:
        for name in ("Test.csv", "test.csv"):
            csv_path = dataset_root / name
            if csv_path.exists():
                test_counts = _count_test_csv(csv_path)
                break

    print(f"Dataset: {dataset_root}")
    _print_counts("Train", train_counts)
    _print_counts("Test", test_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
