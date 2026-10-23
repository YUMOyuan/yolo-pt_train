# 红绿灯分类训练说明

当前数据包含 497 张唯一 JPG 图片，原始目录类别是 `红灯` 和 `绿灯`。数据没有边界框标注，因此使用图像分类任务：模型判断整张图片属于绿灯、红灯或没有红绿灯。`no_light` 需要额外采集真实的无红绿灯画面，不能用现有红灯/绿灯图片代替。

## 数据结构

```text
datasets/
├── data.yaml
└── light/
    ├── train/
    │   ├── green_light/
    │   ├── no_light/
    │   └── red_light/
    └── val/
        ├── green_light/
        ├── no_light/
        └── red_light/
```

`datasets/data.yaml` 内容如下：

```yaml
task: classify
path: light
train: train
val: val

nc: 3
names:
  0: green_light
  1: no_light
  2: red_light
```

分类训练时传入数据集目录或这个 YAML 都可以，训练脚本会自动解析 YAML 中的 `path`。

## 补充 no_light 数据

从摄像头或原始视频中采集没有红绿灯的画面，放入：

```text
datasets/light/train/no_light/
datasets/light/val/no_light/
```

训练集和验证集应来自不同的视频片段或序列，不能只复制同一张图片。建议覆盖不同路口、距离、角度、亮度、天气、车辆遮挡和夜间场景。验证集至少保留 20% 的图片。

## 训练

`yolo11n-cls.pt` 是轻量预训练分类模型，适合当前数据规模。首次运行会自动下载权重。

```bash
python train.py --data datasets/data.yaml --model yolo11n-cls.pt --epochs 100 --imgsz 320 --batch 32 --device 0
```

没有 CUDA 时使用 `--device cpu`，显存不足时把 `--batch` 改成 `16` 或 `8`。最佳权重默认位于：

```text
runs/classify/traffic_light_yolo11n_cls/weights/best.pt
```

## 导出部署模型

```bash
python train.py --data datasets/data.yaml --model runs/classify/traffic_light_yolo11n_cls/weights/best.pt --device 0 --export onnx
```

## 摄像头实时预测

使用 [predict_camera.py](predict_camera.py) 读取电脑摄像头。默认加载本次训练的 `best.pt`，按 `Q` 或 `Esc` 退出，按 `S` 保存当前画面。

```powershell
& "D:\Program Files\anaconda\envs\yolo\python.exe" predict_camera.py --device 0
```

如果摄像头不是默认编号 `0`：

```powershell
& "D:\Program Files\anaconda\envs\yolo\python.exe" predict_camera.py --camera 1 --device 0
```

`--conf 0.60` 是最低置信度，`--smooth 5` 表示使用最近 5 个可信帧进行状态平滑。预测窗口显示 `green_light`、`red_light` 或 `no_light`。
