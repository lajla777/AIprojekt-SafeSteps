import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import config
from src.models.tof.predict import LABEL_NAMES

TOF_MODEL_PATH = SRC_ROOT / 'models' / 'tof' / 'best_model.pth'

LABEL_TEXT = {
    'no_obstacle': 'ni ovire',
    'obstacle_left': 'ovira levo',
    'obstacle_right': 'ovira desno',
    'obstacle_center': 'ovira spredaj',
    'obstacle_right_left': 'ovira levo in desno',
}
_model = None
_model_path: Path | None = None


def get_tof_model():
    global _model, _model_path

    model_path = Path(config.tof_model)
    if not model_path.exists():
        raise FileNotFoundError(f'ToF model not found: {model_path}')

    if _model is None or _model_path != model_path:
        import torch

        from src.models.tof.model import ObstacleCNN

        model = ObstacleCNN(num_classes=len(LABEL_NAMES))
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        _model = model
        _model_path = model_path

    return _model


def normalize_angle_distance(angle_distance: Any):
    import numpy as np

    ad = np.asarray(angle_distance, dtype=np.float32)
    if ad.ndim != 2 or ad.shape[1] != 2:
        raise ValueError('ToF input must have shape (N, 2): [angle, distance]')

    ad = np.nan_to_num(ad, nan=0.0)
    angle_rad = np.deg2rad(ad[:, 0])
    normalized = np.zeros((len(ad), 3), dtype=np.float32)
    normalized[:, 0] = np.sin(angle_rad)
    normalized[:, 1] = np.cos(angle_rad)
    normalized[:, 2] = np.clip(ad[:, 1], 0, 4000) / 4000.0
    return normalized


def predict_angle_distance_window(angle_distance_window: Any) -> dict:
    import numpy as np
    import torch

    model = get_tof_model()
    start = time.perf_counter()

    window = normalize_angle_distance(angle_distance_window)
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    idx = int(np.argmax(probs))
    label = LABEL_NAMES[idx]

    return {
        'label': label,
        'text': LABEL_TEXT.get(label, label),
        'confidence': float(probs[idx]),
        'probabilities': [
            {'label': name, 'text': LABEL_TEXT.get(name, name), 'confidence': float(prob)}
            for name, prob in zip(LABEL_NAMES, probs)
        ],
        'inference_ms': (time.perf_counter() - start) * 1000,
    }


def angle_distance_from_bin(bin_path: str | Path) -> tuple[Any, float]:
    from src.decoder.decode import decode_file

    packets = decode_file(str(bin_path))
    return angle_distance_from_packets(packets)


def angle_distance_from_packets(packets: list[dict]) -> tuple[Any, float]:
    import numpy as np

    from src.tools.label_tool import calibrate_mag, orientation, pair_angle_distance
    from src.tools.visualization import sestavi_podatke, v_pakete

    packets = [_normalize_packet(packet) for packet in packets]
    seznam = v_pakete(packets)
    tof_fvz, tof_matrix = sestavi_podatke(seznam, 0x05)
    gyro_fvz, gyro = sestavi_podatke(seznam, 0x01)
    _, accel = sestavi_podatke(seznam, 0x02)
    _, mag = sestavi_podatke(seznam, 0x03)

    mag = calibrate_mag(mag)
    _, yaw, _, _ = orientation(
        gyro,
        accel,
        mag,
        gyro_fvz=gyro_fvz,
        accel_fvz=gyro_fvz,
        mag_fvz=gyro_fvz,
    )
    yaw = yaw - yaw[0]
    yaw = (yaw + 180) % 360 - 180

    pairs = pair_angle_distance(yaw, tof_matrix, gyro_fvz=gyro_fvz, tof_fvz=tof_fvz)
    return np.asarray(pairs, dtype=np.float32), float(tof_fvz)


def _normalize_packet(packet: dict) -> dict:
    chunks = {}
    for key, chunk in packet.get('chunks', {}).items():
        chunk_id = int(key)
        chunks[chunk_id] = {
            'name': chunk.get('name', ''),
            'unit': chunk.get('unit', ''),
            'samples': [tuple(sample) for sample in chunk.get('samples', [])],
        }

    return {
        'packet_counter': packet.get('packet_counter'),
        'timestamp_ms': packet.get('timestamp_ms'),
        'packet_size': packet.get('packet_size'),
        'chunks': chunks,
    }


def predict_bin_file(bin_path: str | Path) -> dict:
    angle_distance, tof_fvz = angle_distance_from_bin(bin_path)
    window_size = max(1, int(config.tof_window_sec * tof_fvz))
    stride = max(1, int(config.tof_stride_sec * tof_fvz))

    if len(angle_distance) < window_size:
        raise ValueError(f'Premalo ToF podatkov: potrebujem {window_size}, dobim {len(angle_distance)}.')

    windows = []
    start = time.perf_counter()
    for window_start in range(0, len(angle_distance) - window_size + 1, stride):
        windows.append(predict_angle_distance_window(angle_distance[window_start : window_start + window_size]))

    counts: dict[str, int] = {}
    for item in windows:
        counts[item['label']] = counts.get(item['label'], 0) + 1

    majority_label = max(counts, key=counts.get)
    majority_windows = [item for item in windows if item['label'] == majority_label]
    confidence = sum(item['confidence'] for item in majority_windows) / len(majority_windows)

    return {
        'label': majority_label,
        'text': LABEL_TEXT.get(majority_label, majority_label),
        'confidence': confidence,
        'windows': windows,
        'window_count': len(windows),
        'sample_count': len(angle_distance),
        'tof_fvz': tof_fvz,
        'inference_ms': (time.perf_counter() - start) * 1000,
    }
