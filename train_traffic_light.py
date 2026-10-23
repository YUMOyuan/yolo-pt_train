from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(
    data="C:/Users/jd/Desktop/YUMOpack/yumo pack/work_file/YOLO/datasets/第一次训练/data.yaml",
    imgsz=960,
    epochs=100,
    batch=4,
    device=0,
    workers=0,
    seed=42,
    project="C:/Users/jd/Desktop/YUMOpack/yumo pack/work_file/YOLO/runs/detect",
    name="traffic_light_yolo11s",
    exist_ok=True,
    pretrained=True,
    verbose=True,
)
