import threading
from camera import camera_loop
from serial_live import serial_live_loop
from state import add_log
from tof_runtime import tof_prediction_loop
from tts import tts_trigger_loop, tts_worker


def start_background_threads() -> None:
    add_log('SafeSteps starting...', 'info')

    threads = [
        threading.Thread(target=serial_live_loop, daemon=True, name='serial-live'),
        threading.Thread(target=tof_prediction_loop, daemon=True, name='tof-model'),
        threading.Thread(target=camera_loop, daemon=True, name='camera'),
        threading.Thread(target=tts_worker, daemon=True, name='tts'),
        threading.Thread(target=tts_trigger_loop, daemon=True, name='tts-trigger'),
    ]

    for thread in threads:
        thread.start()

    add_log('Background threads started', 'ok')
