# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run the traffic-light classifier on a local webcam."""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_MODEL = Path("runs/classify/runs/classify/traffic_light_yolo11n_cls_v2/weights/best.pt")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Classify traffic-light state from a webcam")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the trained classification model")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--device", default="0", help='Inference device, for example "0" or "cpu"')
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.60, help="Minimum confidence to accept a prediction")
    parser.add_argument("--smooth", type=int, default=5, help="Number of recent accepted predictions to smooth")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the preview")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/predict_camera"), help="Directory for screenshots")
    return parser.parse_args()


def draw_status(frame, label: str, confidence: float, fps: float, history_size: int) -> None:
    """Draw the current classifier status without relying on non-ASCII fonts."""
    color = (0, 200, 0) if label == "green_light" else (0, 0, 220) if label == "red_light" else (0, 180, 220)
    text = f"{label}  {confidence:.1%}" if label != "uncertain" else "uncertain"
    cv2.rectangle(frame, (15, 15), (440, 105), (20, 20, 20), -1)
    cv2.putText(frame, text, (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.1f}  smoothing {history_size}", (30, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q/Esc: quit   S: screenshot", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)


def main() -> None:
    """Open the webcam and classify each frame."""
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if args.smooth < 1:
        raise ValueError("--smooth must be at least 1")

    model = YOLO(str(model_path))
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}. Try --camera 1 or close other camera apps.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    history: deque[str] = deque(maxlen=args.smooth)
    previous_time = time.perf_counter()
    fps = 0.0
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Camera {args.camera} opened. Model: {model_path}")
    print("Press Q or Esc to quit; press S to save a screenshot.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            result = model.predict(source=frame, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            top1 = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            predicted = str(model.names[top1])
            if confidence >= args.conf:
                history.append(predicted)
            stable_label = Counter(history).most_common(1)[0][0] if history else "uncertain"
            stable_confidence = confidence if stable_label == predicted else 0.0

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            fps = instant_fps if fps == 0.0 else 0.85 * fps + 0.15 * instant_fps
            previous_time = now
            draw_status(frame, stable_label, stable_confidence, fps, len(history))
            cv2.imshow("Traffic Light Classifier", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                output = args.save_dir / f"frame_{stamp}.jpg"
                cv2.imwrite(str(output), frame)
                print(f"Saved screenshot: {output}")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
