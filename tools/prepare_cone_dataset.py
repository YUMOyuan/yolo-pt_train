# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Prepare a folder-based image classification dataset for Ultralytics YOLO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
CLASS_ALIASES = {
    "red_light": {"red", "red_light", "redlight", "红灯"},
    "green_light": {"green", "green_light", "greenlight", "绿灯"},
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare a YOLO classification dataset from class folders or a 7z archive")
    parser.add_argument("--source", type=Path, required=True, help="Input directory or .7z archive")
    parser.add_argument("--output", type=Path, default=Path("datasets/cone_light"), help="Output dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio per class, split by filename prefix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used to choose validation groups")
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
    return parser.parse_args()


def find_extractor() -> str:
    """Find a command capable of extracting a 7z archive."""
    for command in ("7z", "7zz", "7za", "bsdtar"):
        if shutil.which(command):
            return command
    raise RuntimeError("No archive extractor found. Install 7-Zip or bsdtar, then run again.")


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract an archive into a temporary directory."""
    command = find_extractor()
    if Path(command).name.lower() == "bsdtar.exe" or command == "bsdtar":
        args = [command, "-xf", str(archive), "-C", str(destination)]
    else:
        args = [command, "x", "-y", f"-o{destination}", str(archive)]
    subprocess.run(args, check=True)


def locate_class_dirs(root: Path) -> dict[str, Path]:
    """Locate one directory for each expected class."""
    found: dict[str, list[Path]] = defaultdict(list)
    for path in (root, *root.rglob("*")):
        if not path.is_dir():
            continue
        name = path.name.strip().lower()
        for class_name, aliases in CLASS_ALIASES.items():
            if name in aliases:
                found[class_name].append(path)
    missing = sorted(set(CLASS_ALIASES) - set(found))
    if missing:
        raise FileNotFoundError(f"Could not find class directories: {', '.join(missing)}")
    duplicates = {name: paths for name, paths in found.items() if len(paths) > 1}
    if duplicates:
        details = "; ".join(f"{name}: {paths}" for name, paths in duplicates.items())
        raise RuntimeError(f"Found multiple directories for one class; use a source with one class pair: {details}")
    return {name: paths[0] for name, paths in found.items()}


def image_records(class_dirs: dict[str, Path]) -> list[dict[str, object]]:
    """Validate images and return records with stable group and hash fields."""
    records: list[dict[str, object]] = []
    seen: dict[str, str] = {}
    for class_name, directory in sorted(class_dirs.items()):
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as error:
                raise ValueError(f"Invalid image: {path} ({error})") from error
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            previous_class = seen.get(digest)
            if previous_class and previous_class != class_name:
                raise ValueError(f"The same image appears in both classes: {path}")
            if previous_class:
                continue
            seen[digest] = class_name
            records.append(
                {
                    "source": path,
                    "class_name": class_name,
                    "group": path.stem.split("_", 1)[0],
                    "sha256": digest,
                }
            )
    if not records:
        raise FileNotFoundError("No supported images were found in the class directories")
    return records


def split_records(records: list[dict[str, object]], val_ratio: float, seed: int) -> None:
    """Assign train/val splits without putting one sequence in both splits."""
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    by_class: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_class[str(record["class_name"])][str(record["group"])].append(record)

    rng = random.Random(seed)
    for class_name, groups in by_class.items():
        group_names = sorted(groups)
        if len(group_names) < 2:
            raise ValueError(f"Class {class_name} needs at least two filename groups for train/val splitting")
        target = sum(len(group) for group in groups.values()) * val_ratio
        candidates = []
        for size in range(1, len(group_names)):
            for combination in itertools.combinations(group_names, size):
                count = sum(len(groups[name]) for name in combination)
                candidates.append((abs(count - target), combination))
        rng.shuffle(candidates)
        val_groups = set(min(candidates, key=lambda candidate: candidate[0])[1])
        for group_name, group_records in groups.items():
            split = "val" if group_name in val_groups else "train"
            for record in group_records:
                record["split"] = split


def copy_dataset(records: list[dict[str, object]], output: Path) -> None:
    """Copy images and write a reproducible manifest."""
    output.mkdir(parents=True)
    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=["split", "class_name", "group", "source", "output", "sha256"])
        writer.writeheader()
        for record in records:
            source = Path(record["source"])
            split = str(record["split"])
            class_name = str(record["class_name"])
            destination_dir = output / split / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / source.name
            if destination.exists():
                destination = destination_dir / f"{source.stem}_{str(record['sha256'])[:8]}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            writer.writerow(
                {
                    "split": split,
                    "class_name": class_name,
                    "group": record["group"],
                    "source": str(source),
                    "output": str(destination),
                    "sha256": record["sha256"],
                }
            )

    (output / "classes.txt").write_text("green_light\nred_light\n", encoding="utf-8")


def main() -> None:
    """Prepare the dataset."""
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output}. Add --force to rebuild it.")
        shutil.rmtree(output)

    if source.is_file() and source.suffix.lower() == ".7z":
        with tempfile.TemporaryDirectory(prefix="cone_dataset_") as temporary_dir:
            temporary_root = Path(temporary_dir)
            extract_archive(source, temporary_root)
            class_dirs = locate_class_dirs(temporary_root)
            records = image_records(class_dirs)
            split_records(records, args.val_ratio, args.seed)
            copy_dataset(records, output)
    elif source.is_dir():
        class_dirs = locate_class_dirs(source)
        records = image_records(class_dirs)
        split_records(records, args.val_ratio, args.seed)
        copy_dataset(records, output)
    else:
        raise FileNotFoundError(f"Source does not exist or is not a supported input: {source}")

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        counts[(str(record["split"]), str(record["class_name"]))] += 1
    print(f"Prepared {len(records)} unique images at {output}")
    for split in ("train", "val"):
        print(f"{split}: " + ", ".join(f"{class_name}={counts[(split, class_name)]}" for class_name in sorted(CLASS_ALIASES)))
    print("Split groups are disjoint; use manifest.csv to audit the source files.")


if __name__ == "__main__":
    main()
