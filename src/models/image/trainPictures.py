from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = YOLO(str(BASE_DIR / "yolov8s.pt"))

model.train(
    data=str(BASE_DIR / "data.yaml"),
    epochs=120,
    imgsz=832,
    batch=8,
    patience=25
)