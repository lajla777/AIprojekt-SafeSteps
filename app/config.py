import sys
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.image.predict import MODEL_PATH


def _path_from_env(name: str) -> Path | None:
    value = os.getenv(name)
    if not value:
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _first_matching(directory: Path, patterns: list[str]) -> Path | None:
    if not directory.exists():
        return None

    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def _model_path(env_name: str, candidates: list[Path], search_dir: Path, patterns: list[str]) -> str:
    env_path = _path_from_env(env_name)
    if env_path is not None:
        return str(env_path)

    found = _first_matching(search_dir, patterns)
    if found is not None:
        return str(found)

    return str(_first_existing(candidates))


IMAGE_MODEL_PATH = _model_path(
    'SAFESTEPS_IMAGE_MODEL',
    [
        MODEL_PATH,
        PROJECT_ROOT / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt',
        PROJECT_ROOT / 'yolov8n.pt',
    ],
    PROJECT_ROOT / 'runs',
    ['detect/train/weights/best.pt', '**/weights/best.pt', '**/*.pt'],
)

TOF_MODEL_PATH = _model_path(
    'SAFESTEPS_TOF_MODEL',
    [
        PROJECT_ROOT / 'src' / 'models' / 'tof' / 'best_model.pth',
    ],
    PROJECT_ROOT / 'src' / 'models' / 'tof',
    ['best_model.pth', '*.pth'],
)

TTS_MODEL_PATH = _model_path(
    'SAFESTEPS_TTS_MODEL',
    [
        PROJECT_ROOT / 'src' / 'models' / 'tts' / 'sl_SI-artur-medium.onnx',
    ],
    PROJECT_ROOT / 'src' / 'models' / 'tts',
    ['*.onnx'],
)


@dataclass
class Config:
    stm32_host: str = '192.168.1.50'
    stm32_port: int = 5000
    stm_service_host: str = '127.0.0.1'
    stm_service_port: int = 5000
    stm32_vid: str = '0483'
    stm32_pid: str = '5740'
    stm32_baudrate: int = 115200
    stm32_serial_port: str = ''
    camera_index: int = 0
    yolo_model: str = IMAGE_MODEL_PATH
    yolo_confidence: float = 0.30
    tof_model: str = TOF_MODEL_PATH
    tof_window_sec: float = 1.0
    tof_stride_sec: float = 0.25
    tof_live_capture_sec: float = 2.0
    tof_sweep_min_angle_deg: float = 15.0
    tof_sweep_prominence_deg: float = 15.0
    tof_sweep_min_samples: int = 25
    tof_min_obstacle_confidence: float = 0.60
    tof_no_obstacle_distance_mm: float = 1800.0
    tof_no_obstacle_valid_ratio: float = 0.15
    tof_mirror_left_right: bool = True
    tof_use_geometry_fallback: bool = True
    tof_center_angle_deg: float = 10.0
    warn_dist: float = 0.80
    danger_dist: float = 0.30
    tts_model: str = TTS_MODEL_PATH
    tts_enabled: bool = True


config = Config()
