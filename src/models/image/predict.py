from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"

_model = None


def get_model():
    global _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Image model not found: {MODEL_PATH}")

    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(str(MODEL_PATH))

    return _model


def predict_source(source, conf: float = 0.30):
    model = get_model()
    start = time.perf_counter()
    
    results = model.predict(
        source,
        verbose=False,
        conf=conf,
    )

    return results, model.names, (time.perf_counter() - start) * 1000


if __name__ == "__main__":
    model = get_model()
    model.predict(
        source="test.jpg",
        show=True,
        save=True
    )
