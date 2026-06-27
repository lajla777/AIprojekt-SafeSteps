import logging
import time
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('safesteps')


@dataclass
class AppState:
    tof_distance: float = -1.0
    tof_last_raw: str = ''
    tof_packets: int = 0
    tof_last_seen: float = 0.0
    tof_search_enabled: bool = False
    tof_angle_distance_window: list[list[float]] = field(default_factory=list)
    tof_live_packets: list[dict] = field(default_factory=list)
    tof_live_bytes: int = 0
    tof_live_raw_packets: int = 0
    tof_live_bad_packets: int = 0
    tof_live_port: str = ''
    tof_prediction: str = ''
    tof_prediction_text: str = ''
    tof_confidence: float = 0.0
    tof_inference_ms: float = 0.0
    tof_probabilities: list[dict] = field(default_factory=list)
    tof_file_result: dict = field(default_factory=dict)

    detections: list[dict] = field(default_factory=list)
    inference_ms: float = 0.0
    camera_enabled: bool = False
    cam_connected: bool = False

    uploaded_image_data_url: str = ''
    uploaded_detections: list[dict] = field(default_factory=list)
    uploaded_inference_ms: float = 0.0

    stm_connected: bool = False
    tts_last: str = ''
    tts_active: bool = False
    last_tts_trigger: float = 0.0
    log_entries: list[dict] = field(default_factory=list)


state = AppState()


def add_log(msg: str, level: str = 'info') -> None:
    entry = {'time': time.strftime('%H:%M:%S'), 'msg': msg, 'level': level}
    state.log_entries.append(entry)
    if len(state.log_entries) > 150:
        state.log_entries.pop(0)
    log.info('[%s] %s', level, msg)
