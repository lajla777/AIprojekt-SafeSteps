from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train-10" / "weights" / "best.pt"

_model = None
_model_path = None


def get_model(model_path: str | Path = MODEL_PATH):
    global _model, _model_path

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Image model not found: {model_path}")

    if _model is None or _model_path != model_path:
        from ultralytics import YOLO

        _model = YOLO(str(model_path))
        _model_path = model_path

    return _model


def predict_source(source, conf: float = 0.30, model_path: str | Path = MODEL_PATH):
    model = get_model(model_path)
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
