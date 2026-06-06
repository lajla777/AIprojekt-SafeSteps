from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data.yaml",
    epochs=120,
    imgsz=832,
    batch=8,
    patience=25
)