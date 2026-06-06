import time
from pathlib import Path
from nicegui import ui
from config import config
from image_utils import results_to_detections
from src.models.image.predict import get_model, predict_source
from state import add_log, state
from tts import speak


def set_camera_enabled(enabled: bool) -> None:
    if enabled and not Path(config.yolo_model).exists():
        add_log(f'Image model is missing: {config.yolo_model}', 'danger')
        ui.notify(f'Manjka model: {config.yolo_model}', type='negative')
        return
    state.camera_enabled = enabled
    add_log('Camera requested by user' if enabled else 'Camera disabled by user', 'info')
    speak('Kamera je vklopljena.' if enabled else 'Kamera je izklopljena.')


def camera_loop() -> None:
    try:
        import cv2
    except ImportError:
        add_log('opencv is not installed', 'danger')
        return

    model = None
    cap = None

    while True:
        if not state.camera_enabled:
            if cap is not None:
                cap.release()
                cap = None
                state.cam_connected = False
                add_log('Camera released', 'info')

            time.sleep(0.2)
            continue

        if model is None:
            try:
                model = get_model()
                add_log(f'YOLO model loaded: {config.yolo_model}', 'ok')
            except Exception as exc:
                add_log(f'YOLO load error: {exc}', 'danger')
                state.camera_enabled = False
                time.sleep(1)
                continue

        if cap is None:
            cap = cv2.VideoCapture(config.camera_index)
            if not cap.isOpened():
                add_log('Camera is not accessible', 'danger')
                state.cam_connected = False
                state.camera_enabled = False
                time.sleep(1)
                continue

            state.cam_connected = True
            add_log('Camera active', 'ok')

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        results, names, state.inference_ms = predict_source(
            frame,
            conf=config.yolo_confidence,
        )
        state.detections = results_to_detections(results, names, config.yolo_confidence)
        time.sleep(0.1)
