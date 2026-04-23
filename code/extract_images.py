#!/usr/bin/env python3
"""Extract image assets from official KownledgeBase.zip.

Default behavior:
- Read images under zip folder containing "插图"
- Extract to data/images with flat names to simplify lookup
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract images from KownledgeBase.zip")
    parser.add_argument("--zip-file", type=Path, default=Path("data/KownledgeBase.zip"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/images"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.zip_file.exists():
        raise FileNotFoundError(f"zip not found: {args.zip_file}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with zipfile.ZipFile(args.zip_file, "r") as zf:
        for name in zf.namelist():
            lname = name.lower()
            if "插图" not in name:
                continue
            if not lname.endswith(IMAGE_SUFFIXES):
                continue

            # Use filename only to keep lookup simple.
            target = args.out_dir / Path(name).name
            with zf.open(name) as src, target.open("wb") as dst:
                dst.write(src.read())
            extracted += 1

    print(f"extracted={extracted}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
