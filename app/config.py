import sys
import os
from dataclasses import dataclass
from pathlib import Path


def _app_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _bundle_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _app_root()
BUNDLE_ROOT = _bundle_root()

for path in (PROJECT_ROOT, BUNDLE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


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
        matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def _model_path(env_name: str, candidates: list[Path], search_dirs: list[Path], patterns: list[str]) -> str:
    env_path = _path_from_env(env_name)
    if env_path is not None:
        return str(env_path)

    for search_dir in search_dirs:
        found = _first_matching(search_dir, patterns)
        if found is not None:
            return str(found)

    return str(_first_existing(candidates))


IMAGE_MODEL_PATH = _model_path(
    'SAFESTEPS_IMAGE_MODEL',
    [
        BUNDLE_ROOT / 'runs' / 'detect' / 'train-10' / 'weights' / 'best.pt',
        PROJECT_ROOT / 'runs' / 'detect' / 'train-10' / 'weights' / 'best.pt',
        BUNDLE_ROOT / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt',
        PROJECT_ROOT / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt',
        BUNDLE_ROOT / 'src' / 'models' / 'image' / 'yolov8n.pt',
        PROJECT_ROOT / 'src' / 'models' / 'image' / 'yolov8n.pt',
    ],
    [
        BUNDLE_ROOT / 'runs',
        PROJECT_ROOT / 'runs',
        BUNDLE_ROOT / 'src' / 'models' / 'image',
        PROJECT_ROOT / 'src' / 'models' / 'image',
    ],
    ['detect/train-10/weights/best.pt', 'detect/train-*/weights/best.pt', 'detect/train/weights/best.pt', '**/weights/best.pt', '**/*.pt'],
)

TOF_MODEL_PATH = _model_path(
    'SAFESTEPS_TOF_MODEL',
    [
        BUNDLE_ROOT / 'src' / 'models' / 'tof' / 'best_model.pth',
        PROJECT_ROOT / 'src' / 'models' / 'tof' / 'best_model.pth',
    ],
    [
        BUNDLE_ROOT / 'src' / 'models' / 'tof',
        PROJECT_ROOT / 'src' / 'models' / 'tof',
    ],
    ['best_model.pth', '*.pth'],
)

TTS_MODEL_PATH = _model_path(
    'SAFESTEPS_TTS_MODEL',
    [
        BUNDLE_ROOT / 'src' / 'models' / 'tts' / 'sl_SI-artur-medium.onnx',
        PROJECT_ROOT / 'src' / 'models' / 'tts' / 'sl_SI-artur-medium.onnx',
    ],
    [
        BUNDLE_ROOT / 'src' / 'models' / 'tts',
        PROJECT_ROOT / 'src' / 'models' / 'tts',
    ],
    ['*.onnx'],
)


@dataclass
class Config:
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