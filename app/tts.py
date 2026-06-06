import queue
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

from config import config
from state import add_log, state

tts_queue: queue.Queue[str] = queue.Queue()
_last_requested_text = ''
_last_requested_at = 0.0
_piper_voice = None
_piper_model_path: Path | None = None


def speak(msg: str, min_interval: float = 0.0) -> None:
    global _last_requested_text, _last_requested_at

    if config.tts_enabled and msg:
        now = time.time()
        if msg == _last_requested_text and now - _last_requested_at < min_interval:
            return

        _last_requested_text = msg
        _last_requested_at = now
        tts_queue.put(msg)


def tts_worker() -> None:
    while True:
        msg = tts_queue.get()
        if not msg or not config.tts_enabled:
            continue

        state.tts_active = True
        try:
            speak_with_piper(msg)
        except Exception as exc:
            add_log(f'Piper TTS error: {exc}', 'warn')
        finally:
            state.tts_active = False


def get_piper_voice():
    global _piper_voice, _piper_model_path

    model_path = Path(config.tts_model)
    if not model_path.exists():
        raise FileNotFoundError(f'Piper voice model not found: {model_path}')

    if _piper_voice is None or _piper_model_path != model_path:
        from piper.voice import PiperVoice

        _piper_voice = PiperVoice.load(str(model_path))
        _piper_model_path = model_path
        add_log(f'Piper TTS model loaded: {model_path.name}', 'ok')

    return _piper_voice


def speak_with_piper(msg: str) -> None:
    voice = get_piper_voice()

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as file:
        wav_path = Path(file.name)

    try:
        with wave.open(str(wav_path), 'wb') as wav_file:
            _synthesize_piper_chunks(voice, msg, wav_file)

        _play_wav(wav_path)
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except OSError:
            pass


def _synthesize_piper_chunks(voice, msg: str, wav_file) -> None:
    sample_rate = _piper_sample_rate(voice)
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)

    for chunk in voice.synthesize(msg):
        wav_file.writeframes(chunk.audio_int16_bytes)


def _piper_sample_rate(voice) -> int:
    voice_config = getattr(voice, 'config', None)
    sample_rate = getattr(voice_config, 'sample_rate', None)
    if sample_rate is None and isinstance(voice_config, dict):
        sample_rate = voice_config.get('sample_rate')
    return int(sample_rate or 22050)


def _play_wav(path: Path) -> None:
    if sys.platform.startswith('win'):
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return

    for command in (['aplay', str(path)], ['paplay', str(path)], ['afplay', str(path)]):
        try:
            completed = subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if completed.returncode == 0:
                return
        except FileNotFoundError:
            continue

    raise RuntimeError('No WAV player found for Piper output.')


def build_tts_message() -> Optional[str]:
    distance = state.tof_distance

    if distance < 0:
        return None

    if distance < config.danger_dist:
        urgency = 'Nevarnost!'
    elif distance < config.warn_dist:
        urgency = 'Pozor.'
    else:
        return None

    dist_cm = int(distance * 100)

    if state.detections:
        labels = ', '.join(item.get('text', item['label']) for item in state.detections[:2])
        return f'{urgency} {labels}, razdalja {dist_cm} centimetrov.'

    return f'{urgency} Ovira, razdalja {dist_cm} centimetrov.'


def tts_trigger_loop() -> None:
    while True:
        msg = build_tts_message()

        if msg and msg != state.tts_last:
            now = time.time()
            if now - state.last_tts_trigger > 2.0:
                state.tts_last = msg
                state.last_tts_trigger = now
                speak(msg)
                add_log(f'TTS: "{msg}"', 'info')

        time.sleep(0.5)
