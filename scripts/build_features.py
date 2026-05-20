"""Build the DINOv3 (or DINOv2) feature cache for every labeled image.

Heavy step — run once locally (M-series MPS / CUDA), then every downstream
pipeline (DINOv3 head, k-NN, etc.) reads the parquet.

Reads the union of `data/processed/train/attr_*_train.csv` image paths so
every image referenced by any per-attribute training file is embedded.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --model facebook/dinov2-large
    python scripts/build_features.py --max-images 64  # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.embed import (  # noqa: E402
    DEFAULT_DINOV3_MODEL,
    extract_features_for_split,
    save_features,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_DINOV3_MODEL)
    p.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/processed/train"),
        help="Directory holding per-attribute attr_*_train.csv files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
        help="Parquet output directory (gitignored).",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=[None, "fp16", "bf16"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-images", type=int, default=None)
    return p.parse_args()


def collect_image_paths(train_dir: Path) -> pd.DataFrame:
    """Union of image_paths across all attr_*_train.csv files, deduplicated."""
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(train_dir.glob("attr_*_train.csv")):
        df = pd.read_csv(csv_path, usecols=["image_path"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No attr_*_train.csv files found in {train_dir}. "
            "Make sure data/processed/train/ is populated."
        )
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="image_path")
    return combined.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    combined = collect_image_paths(args.train_dir)
    print(f"Will encode {len(combined)} unique images from {args.train_dir}")

    cache = extract_features_for_split(
        combined,
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_images=args.max_images,
    )

    print(f"Encoded {len(cache.df)} images into {cache.dim}-d embeddings.")
    p = save_features(cache, out_dir=args.output_dir)
    print(f"Wrote feature cache to {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
