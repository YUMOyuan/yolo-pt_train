# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train, validate, and optionally export the prepared light-state classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a lightweight red/green image classifier")
    parser.add_argument("--model", default="yolo11n-cls.pt", help="Classification checkpoint or YAML model")
    parser.add_argument(
        "--data", type=Path, default=Path("datasets/data.yaml"), help="Classification dataset directory or YAML config"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--imgsz", type=int, default=320, help="Square training image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size; reduce it when GPU memory is limited")
    parser.add_argument("--device", default=None, help='Device, for example "0" or "cpu"')
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--project", default="runs/classify", help="Training result directory")
    parser.add_argument("--name", default="traffic_light_yolo11n_cls", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--export", choices=("onnx", "openvino", "torchscript"), help="Export the best model after validation"
    )
    return parser.parse_args()


def resolve_data_path(data_arg: Path) -> Path:
    """Resolve a classification directory from a directory or project YAML file."""
    data_arg = data_arg.expanduser().resolve()
    if data_arg.suffix.lower() in {".yaml", ".yml"}:
        config = YAML.load(data_arg)
        root = Path(config.get("path", "."))
        data_arg = (data_arg.parent / root if not root.is_absolute() else root).resolve()
    return data_arg


def main() -> None:
    """Train, validate, and optionally export the classifier."""
    args = parse_args()
    data = resolve_data_path(args.data)
    if not (data / "train").is_dir() or not (data / "val").is_dir():
        raise FileNotFoundError(
            f"Expected classification folders at {data}/train and {data}/val. Run the preparation script first."
        )

    train_args = {
        "data": str(data),
        "task": "classify",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "pretrained": True,
    }
    if args.device is not None:
        train_args["device"] = args.device

    model = YOLO(args.model)
    model.train(**train_args)
    best = Path(model.trainer.best) if getattr(model, "trainer", None) else None
    best_model = YOLO(str(best)) if best and best.exists() else model
    metrics = best_model.val(data=str(data), imgsz=args.imgsz, batch=args.batch, workers=args.workers)
    print(f"Validation top-1 accuracy: {metrics.top1:.4f}")
    print(f"Validation top-5 accuracy: {metrics.top5:.4f}")

    if args.export:
        exported = best_model.export(format=args.export, imgsz=args.imgsz)
        print(f"Exported model: {exported}")


if __name__ == "__main__":
    main()
