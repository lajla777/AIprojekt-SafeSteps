import time

from config import config
from state import add_log, state
from tof_model import angle_distance_from_packets, predict_angle_distance_window

_last_sweep_key = None
_recent_predictions = []
_last_prediction_at = 0.0


def tof_prediction_loop() -> None:
    """Izvaja live ToF napoved v ozadju.

    Koraki:
    - vzame zadnje STM32 pakete iz `state.tof_live_packets`;
    - jih pretvori v pare kot-razdalja;
    - izvede napoved z ToF modelom ali pravilom "ni ovire";
    - stabilizira rezultat in posodobi `state` za GUI/TTS.
    """
    global _last_sweep_key, _recent_predictions, _last_prediction_at

    while True:
        if state.tof_search_enabled:
            try:
                packets = list(state.tof_live_packets[-220:])
                packet_count = len(state.tof_live_packets)

                if len(packets) < 3:
                    state.tof_last_raw = (
                        f'{len(packets)} paketov, '
                        f'{state.tof_live_bytes} bytes, '
                        f'{state.tof_live_bad_packets} slabih'
                    )
                    time.sleep(0.5)
                    continue

                angle_distance, tof_fvz = angle_distance_from_packets(packets)
                distance_stats = _distance_stats(angle_distance)
                if distance_stats['min_mm'] != float('inf'):
                    state.tof_distance = distance_stats['min_mm'] / 1000.0
                else:
                    state.tof_distance = -1.0

                result = (
                    _forced_no_obstacle(distance_stats)
                    if _looks_like_no_obstacle(distance_stats)
                    else _predict_signal_windows(angle_distance, tof_fvz)
                )

                if result is None:
                    state.tof_last_raw = (
                        f'{len(angle_distance)} vzorcev, '
                        f'veljavne razdalje {distance_stats["valid_ratio"]:.0%}; '
                        f'čakamo dovolj podatkov za 1s okno'
                    )
                    time.sleep(0.5)
                    continue

                signal_key = _signal_key(angle_distance)

                if signal_key == _last_sweep_key:
                    time.sleep(0.3)
                    continue

                now = time.time()
                if now - _last_prediction_at < 1.2:
                    time.sleep(0.3)
                    continue

                _last_sweep_key = signal_key
                _last_prediction_at = now
                result = _apply_live_label_correction(result)

                if result['label'] == 'no_obstacle':
                    _recent_predictions = []

                _recent_predictions.append(result)
                _recent_predictions = _recent_predictions[-5:]

                stable_result = _stable_prediction(_recent_predictions)
                stable_result = _apply_distance_and_confidence_gate(stable_result, angle_distance)
                stable_result = _apply_geometry_fallback(stable_result, angle_distance)
                state.stm_connected = True
                state.tof_prediction = stable_result['label']
                state.tof_prediction_text = stable_result['text']
                state.tof_confidence = stable_result['confidence']
                state.tof_inference_ms = result['inference_ms']
                state.tof_probabilities = stable_result['probabilities']
                state.tof_packets = packet_count
                state.tof_last_seen = time.time()
                state.tof_last_raw = (
                    f'{len(angle_distance)} vzorcev, '
                    f'{result.get("window_count", 1)} oken, '
                    f'{tof_fvz:.1f} Hz, '
                    f'veljavne razdalje {distance_stats["valid_ratio"]:.0%}, '
                    f'min {distance_stats["min_mm"]:.0f} mm, '
                    f'mediana {distance_stats["median_mm"]:.0f} mm, '
                    f'kot ovire {stable_result.get("obstacle_angle_deg", float("nan")):.0f} stopinj, '
                    f'{state.tof_live_bytes} bytes; '
                    f'zadnja surova napoved: {result["text"]} ({int(result["confidence"] * 100)}%)'
                )
                log_level = 'ok' if stable_result['stable'] else 'warn'
                add_log(
                    f'Live ToF stable prediction: {stable_result["text"]} '
                    f'({int(stable_result["confidence"] * 100)}%, raw {result["text"]}, '
                    f'{stable_result["vote_count"]}/{stable_result["history_count"]} glasov, '
                    f'veljavne razdalje {distance_stats["valid_ratio"]:.0%}, '
                    f'{stable_result.get("min_distance_mm", 0):.0f} mm min, '
                    f'{stable_result.get("obstacle_angle_deg", float("nan")):.0f} deg)',
                    log_level,
                )
            except Exception as exc:
                state.tof_prediction = ''
                state.tof_prediction_text = ''
                state.tof_confidence = 0.0
                state.tof_last_raw = f'Napaka pri live ToF napovedi: {exc}'
                add_log(f'Live ToF prediction failed: {exc}', 'warn')
                time.sleep(1)

        time.sleep(0.5)


def _predict_signal_windows(angle_distance, tof_fvz: float) -> dict | None:
    """Napove razred iz več prekrivajočih se oken.

    - dolžina okna je določena z `config.tof_window_sec`;
    - razmik med okni je določen z `config.tof_stride_sec`;
    - končni razred dobi iz povprečja verjetnosti vseh oken.
    """
    window_size = max(1, int(config.tof_window_sec * tof_fvz))
    stride = max(1, int(config.tof_stride_sec * tof_fvz))

    if len(angle_distance) < window_size:
        return None

    windows = []
    for start in range(0, len(angle_distance) - window_size + 1, stride):
        windows.append(predict_angle_distance_window(angle_distance[start:start + window_size]))

    if not windows:
        return None

    label_order = [item['label'] for item in windows[0]['probabilities']]
    mean_probs = []
    for label in label_order:
        values = []
        for item in windows:
            values.extend(prob['confidence'] for prob in item['probabilities'] if prob['label'] == label)
        mean_probs.append(sum(values) / len(values))

    best_idx = max(range(len(mean_probs)), key=lambda idx: mean_probs[idx])
    label = label_order[best_idx]
    template = windows[-1]
    result = dict(template)
    result['label'] = label
    result['text'] = next(
        (prob['text'] for prob in template['probabilities'] if prob['label'] == label),
        label,
    )
    result['confidence'] = float(mean_probs[best_idx])
    result['probabilities'] = [
        {
            'label': prob['label'],
            'text': prob['text'],
            'confidence': float(mean_probs[idx]),
        }
        for idx, prob in enumerate(template['probabilities'])
    ]
    result['window_count'] = len(windows)
    return result


def _distance_stats(angle_distance) -> dict:
    """Izračuna osnovne statistike razdalj za pravila in TTS."""
    import numpy as np

    distances = np.asarray(angle_distance[:, 1], dtype=float)
    valid = np.isfinite(distances) & (distances > 0) & (distances < 4000)
    valid_values = distances[valid]
    return {
        'count': int(len(distances)),
        'valid_count': int(valid.sum()),
        'valid_ratio': float(valid.sum() / len(distances)) if len(distances) else 0.0,
        'min_mm': float(np.min(valid_values)) if len(valid_values) else float('inf'),
        'median_mm': float(np.median(valid_values)) if len(valid_values) else float('nan'),
    }


def _looks_like_no_obstacle(stats: dict) -> bool:
    """Vrne `True`, če razdalje bolj ustrezajo stanju brez ovire."""
    if stats['count'] < 20:
        return False
    if stats['valid_count'] == 0:
        return True
    if stats['valid_ratio'] < 0.10:
        return True
    return stats['valid_ratio'] < config.tof_no_obstacle_valid_ratio and (
        stats['min_mm'] > config.tof_no_obstacle_distance_mm
        or stats['median_mm'] > 1200
    )


def _forced_no_obstacle(stats: dict) -> dict:
    """Sestavi rezultat `no_obstacle` brez klica nevronske mreže."""
    probabilities = [
        {'label': 'no_obstacle', 'text': 'ni ovire', 'confidence': 0.90},
        {'label': 'obstacle_left', 'text': 'ovira levo', 'confidence': 0.025},
        {'label': 'obstacle_right', 'text': 'ovira desno', 'confidence': 0.025},
        {'label': 'obstacle_center', 'text': 'ovira spredaj', 'confidence': 0.025},
        {'label': 'obstacle_right_left', 'text': 'ovira levo in desno', 'confidence': 0.025},
    ]
    return {
        'label': 'no_obstacle',
        'text': 'ni ovire',
        'confidence': 0.90,
        'probabilities': probabilities,
        'inference_ms': 0.0,
        'window_count': 0,
        'distance_stats': stats,
    }


def _stable_prediction(predictions: list[dict]) -> dict:
    """Stabilizira zadnje live napovedi.

    Uporabi:
    - povprečje verjetnosti po razredih;
    - število glasov za izbrani razred;
    - oznako `stable`, kadar ima rezultat dovolj podpore.
    """
    import numpy as np

    if not predictions:
        raise ValueError('Ni ToF napovedi za stabilizacijo.')

    counts: dict[str, int] = {}
    for item in predictions:
        counts[item['label']] = counts.get(item['label'], 0) + 1

    label_order = [item['label'] for item in predictions[-1]['probabilities']]
    probability_rows = []
    for item in predictions:
        probability_rows.append([prob['confidence'] for prob in item['probabilities']])

    mean_probabilities = np.mean(np.asarray(probability_rows, dtype=float), axis=0)
    best_idx = int(np.argmax(mean_probabilities))
    label = label_order[best_idx]
    vote_count = counts.get(label, 0)
    history_count = len(predictions)
    stable = vote_count >= min(3, history_count) or float(mean_probabilities[best_idx]) >= 0.70

    template = next((item for item in reversed(predictions) if item['label'] == label), predictions[-1])
    result = dict(template)
    result['label'] = label
    result['text'] = next(
        (prob['text'] for prob in template['probabilities'] if prob['label'] == label),
        label,
    )
    result['confidence'] = float(mean_probabilities[best_idx])
    result['probabilities'] = [
        {
            'label': prob['label'],
            'text': prob['text'],
            'confidence': float(mean_probabilities[idx]),
        }
        for idx, prob in enumerate(template['probabilities'])
    ]
    result['vote_count'] = vote_count
    result['history_count'] = history_count
    result['stable'] = stable
    return result


def _apply_live_label_correction(result: dict) -> dict:
    """Po potrebi zrcali razreda levo/desno.

    Uporabi se samo, kadar je `config.tof_mirror_left_right` vklopljen.
    Ostali razredi ostanejo nespremenjeni.
    """
    if not config.tof_mirror_left_right:
        return result

    mirror = {
        'obstacle_left': 'obstacle_right',
        'obstacle_right': 'obstacle_left',
    }
    mirrored = dict(result)
    mirrored['label'] = mirror.get(result['label'], result['label'])
    mirrored['probabilities'] = [_mirror_probability_item(item) for item in result['probabilities']]

    for item in mirrored['probabilities']:
        if item['label'] == mirrored['label']:
            mirrored['text'] = item['text']
            mirrored['confidence'] = item['confidence']
            break

    return mirrored


def _mirror_probability_item(item: dict) -> dict:
    """Zrcali en element iz seznama verjetnosti."""
    label = item['label']
    if label == 'obstacle_left':
        return {**item, 'label': 'obstacle_right', 'text': 'ovira desno'}
    if label == 'obstacle_right':
        return {**item, 'label': 'obstacle_left', 'text': 'ovira levo'}
    return item


def _apply_distance_and_confidence_gate(result: dict, sweep_data) -> dict:
    """Filtrira napovedi ovire z razdaljami in zaupanjem.

    Napoved spremeni v `no_obstacle`, če:
    - je veljavnih razdalj premalo;
    - je zaupanje nizko in ni bližnje razdalje.
    """
    import numpy as np

    gated = dict(result)
    distances = np.asarray(sweep_data[:, 1], dtype=float)
    distances = distances[np.isfinite(distances)]
    distances = distances[distances > 0]

    min_distance = float(np.nanmin(distances)) if len(distances) else float('inf')
    median_distance = float(np.nanmedian(distances)) if len(distances) else float('inf')
    gated['min_distance_mm'] = min_distance
    gated['median_distance_mm'] = median_distance

    is_obstacle = gated['label'] != 'no_obstacle'
    low_confidence = gated['confidence'] < config.tof_min_obstacle_confidence
    sparse_distances = len(distances) < 4
    no_close_distance = min_distance > config.tof_no_obstacle_distance_mm or median_distance > 1200

    if is_obstacle and (sparse_distances or (low_confidence and no_close_distance)):
        gated['label'] = 'no_obstacle'
        gated['text'] = 'ni ovire'
        gated['confidence'] = max(gated['confidence'], 1.0 - gated['confidence'])
        gated['stable'] = False

    return gated


def _apply_geometry_fallback(result: dict, angle_distance) -> dict:
    """Doda geometrijski fallback pri negotovem modelu.

    Geometrija ne nadomešča mreže. Uporabi se samo, če:
    - je zaupanje modela prenizko;
    - model napove `no_obstacle`, podatki pa kažejo bližnjo oviro.
    """
    if not config.tof_use_geometry_fallback:
        return result

    geometry = _geometry_prediction(angle_distance)
    if geometry is None:
        return result

    corrected = dict(result)
    corrected['obstacle_angle_deg'] = geometry['angle_deg']

    model_is_uncertain = corrected['confidence'] < config.tof_min_obstacle_confidence
    model_says_no_obstacle = corrected['label'] == 'no_obstacle'

    if model_is_uncertain or model_says_no_obstacle:
        corrected['label'] = geometry['label']
        corrected['text'] = geometry['text']
        corrected['confidence'] = max(corrected['confidence'], geometry['confidence'])
        corrected['stable'] = False
        corrected['probabilities'] = _probabilities_for_label(geometry['label'], geometry['confidence'])

    return corrected


def _geometry_prediction(angle_distance) -> dict | None:
    """Oceni smer ovire iz bližnjih razdalj.

    - izbere razdalje pod `config.tof_no_obstacle_distance_mm`;
    - izračuna utežen povprečni kot ovire;
    - kot pretvori v razred levo/desno/spredaj.
    """
    import numpy as np

    values = np.asarray(angle_distance, dtype=float)
    angles = values[:, 0]
    distances = values[:, 1]
    valid = np.isfinite(distances) & (distances > 0) & (distances < config.tof_no_obstacle_distance_mm)

    if int(valid.sum()) < 3:
        return None

    close_angles = angles[valid]
    close_distances = distances[valid]
    weights = 1.0 / np.maximum(close_distances, 1.0)
    obstacle_angle = float(np.average(close_angles, weights=weights))

    if abs(obstacle_angle) <= config.tof_center_angle_deg:
        label = 'obstacle_center'
        text = 'ovira spredaj'
    elif obstacle_angle > 0:
        label = 'obstacle_right'
        text = 'ovira desno'
    else:
        label = 'obstacle_left'
        text = 'ovira levo'

    if config.tof_mirror_left_right:
        if label == 'obstacle_left':
            label, text = 'obstacle_right', 'ovira desno'
        elif label == 'obstacle_right':
            label, text = 'obstacle_left', 'ovira levo'

    confidence = min(0.85, 0.45 + float(valid.sum()) / max(len(distances), 1))
    return {
        'label': label,
        'text': text,
        'confidence': confidence,
        'angle_deg': obstacle_angle,
    }


def _probabilities_for_label(label: str, confidence: float) -> list[dict]:
    """Sestavi seznam verjetnosti za rezultat iz pravil ali geometrije."""
    labels = [
        ('no_obstacle', 'ni ovire'),
        ('obstacle_left', 'ovira levo'),
        ('obstacle_right', 'ovira desno'),
        ('obstacle_center', 'ovira spredaj'),
        ('obstacle_right_left', 'ovira levo in desno'),
    ]
    rest = max(0.0, 1.0 - confidence) / (len(labels) - 1)
    return [
        {'label': item_label, 'text': text, 'confidence': confidence if item_label == label else rest}
        for item_label, text in labels
    ]


def _signal_key(angle_distance) -> tuple:
    """Vrne podpis signala za zaznavanje že obdelanih oken."""
    first_angle = round(float(angle_distance[0, 0]), 1)
    last_angle = round(float(angle_distance[-1, 0]), 1)
    return len(angle_distance), first_angle, last_angle
