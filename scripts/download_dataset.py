from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download GTSRB Kaggle dataset via kagglehub.")
    parser.add_argument("--out", type=str, default="data", help="Output folder (will be created).")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import kagglehub

    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    print("KaggleHub downloaded dataset to:", path)
    pointer = out_dir / "dataset_path.txt"
    pointer.write_text(str(path), encoding="utf-8")
    print("Wrote dataset pointer:", pointer)
    print("Tip: pass this directory to training via --data (or read dataset_path.txt).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
