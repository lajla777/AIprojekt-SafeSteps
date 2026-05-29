from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

model.predict(
    source="test.jpg",
    show=True,
    save=True
)