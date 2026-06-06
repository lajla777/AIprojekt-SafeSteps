from ultralytics import YOLO

model = YOLO("runs/detect/train-10/weights/best.pt")

model.predict(
    source="test.jpg",
    show=True,
    save=True
)