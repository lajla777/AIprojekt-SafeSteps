from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = YOLO(str(BASE_DIR / "yolov8n.pt"))

model.train(
    data=str(BASE_DIR / "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=8
)
