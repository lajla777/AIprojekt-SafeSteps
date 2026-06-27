import sys
import threading
import time
from pathlib import Path
from queue import Queue

import serial
import serial.tools.list_ports

from config import config
from state import add_log, state
from tts import speak

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decoder.decode import SYNC, parse_packet

MAX_LIVE_PACKETS = 240

_serial_conn = None
_buffer = bytearray()
_serial_lock = threading.RLock()
SERIAL_OPEN_TIMEOUT = 3.0


def find_stm32_port() -> str:
    if config.stm32_serial_port:
        return config.stm32_serial_port

    wanted_vid = config.stm32_vid.upper()
    wanted_pid = config.stm32_pid.upper()

    for port in serial.tools.list_ports.comports():
        vid = f'{port.vid:04X}' if port.vid is not None else ''
        pid = f'{port.pid:04X}' if port.pid is not None else ''
        hwid = (port.hwid or '').upper()

        if vid == wanted_vid and pid == wanted_pid:
            return port.device
        if f'VID:PID={wanted_vid}:{wanted_pid}' in hwid:
            return port.device

    return ''


def set_serial_live_enabled(enabled: bool) -> bool:
    if enabled:
        ok = _open_serial_and_start_stream()
        state.tof_search_enabled = ok
        if ok:
            add_log('Direct serial live stream started', 'info')
            speak('Live zajem STM podatkov je vklopljen.')
        return ok

    state.tof_search_enabled = False
    _stop_serial_live()
    state.stm_connected = False
    state.tof_prediction = ''
    state.tof_prediction_text = ''
    state.tof_confidence = 0.0
    add_log('Direct serial live stream stopped', 'info')
    speak('Live zajem STM podatkov je ustavljen.')
    return True


def _open_serial_and_start_stream() -> bool:
    global _serial_conn, _buffer

    with _serial_lock:
        add_log('Opening STM32 serial stream...', 'info')
        if _serial_conn is not None and _serial_conn.is_open:
            add_log('STM32 serial already open; sending STREAM again', 'info')
            _send_stream_command(_serial_conn)
            return True

        port = find_stm32_port()
        if not port:
            add_log('STM32 serial port not found', 'danger')
            speak('STM32 naprava ni najdena.')
            return False

        try:
            state.tof_live_port = port
            state.tof_last_raw = f'Odpiram serial port {port}...'
            add_log(f'Opening serial port {port} @ {config.stm32_baudrate}', 'info')
            _serial_conn = _open_serial_with_timeout(port)
            add_log(f'Serial port {port} opened; setting DTR/RTS', 'info')
            _serial_conn.setDTR(True)
            _serial_conn.setRTS(True)
            time.sleep(0.5)
            add_log('Resetting serial buffers', 'info')
            _serial_conn.reset_input_buffer()
            _serial_conn.reset_output_buffer()
        except Exception as exc:
            add_log(f'Cannot open STM32 serial port {port}: {exc}', 'danger')
            speak('STM32 serial port ni dostopen.')
            _serial_conn = None
            state.tof_search_enabled = False
            state.stm_connected = False
            state.tof_last_raw = (
                f'Port {port} ni dostopen. Zapri PuTTY, prejšnjo instanco aplikacije '
                f'ali drug program, ki uporablja STM32.'
            )
            return False

        _buffer = bytearray()
        state.tof_live_packets.clear()
        state.tof_live_bytes = 0
        state.tof_live_raw_packets = 0
        state.tof_live_bad_packets = 0
        state.tof_live_port = port
        state.tof_packets = 0
        state.tof_last_seen = 0.0
        state.tof_last_raw = 'Serial odprt, STREAM poslan, čakam podatke...'
        state.stm_connected = True

        _send_stream_command(_serial_conn)
        add_log(f'STM32 serial opened on {port}', 'ok')
        return True


def _open_serial_with_timeout(port: str):
    result_queue = Queue(maxsize=1)

    def open_worker() -> None:
        try:
            conn = serial.Serial(
                port=port,
                baudrate=config.stm32_baudrate,
                timeout=0.05,
                write_timeout=2,
                rtscts=False,
                dsrdtr=False,
            )
            result_queue.put((conn, None))
        except Exception as exc:
            result_queue.put((None, exc))

    thread = threading.Thread(target=open_worker, daemon=True, name='serial-open')
    thread.start()
    thread.join(SERIAL_OPEN_TIMEOUT)

    if thread.is_alive():
        raise TimeoutError(
            f'Opening {port} timed out after {SERIAL_OPEN_TIMEOUT:.0f}s. '
            'Windows driver ali drug proces drzi COM port.'
        )

    conn, exc = result_queue.get()
    if exc is not None:
        raise exc
    return conn


def _send_stream_command(ser) -> None:
    add_log('Sending STREAM command to STM32', 'info')
    for suffix in ('\r\n', '\n', '\r'):
        ser.write(('STREAM' + suffix).encode('utf-8'))
        ser.flush()
        time.sleep(0.05)
    add_log('Sent STREAM to STM32 over direct serial', 'info')


def _stop_serial_live() -> None:
    global _serial_conn

    with _serial_lock:
        if _serial_conn is None:
            return

        try:
            if _serial_conn.is_open:
                for command in ('STOP', 'STOP_STREAM', 'END'):
                    try:
                        _serial_conn.write((command + '\r\n').encode('utf-8'))
                        _serial_conn.flush()
                        time.sleep(0.03)
                    except serial.SerialException:
                        break
                _serial_conn.close()
        except serial.SerialException:
            pass
        finally:
            _serial_conn = None


def serial_live_loop() -> None:
    last_wait_log = 0.0

    while True:
        if not state.tof_search_enabled:
            time.sleep(0.1)
            continue

        if _serial_conn is None or not _serial_conn.is_open:
            state.tof_search_enabled = False
            state.stm_connected = False
            state.tof_last_raw = 'Serial povezava ni odprta. Ponovno klikni Začni live zajem.'
            time.sleep(0.5)
            continue

        try:
            data = _serial_conn.read(512)
        except serial.SerialException as exc:
            add_log(f'STM32 serial read failed: {exc}', 'warn')
            state.stm_connected = False
            state.tof_search_enabled = False
            _stop_serial_live()
            time.sleep(1)
            continue

        if data:
            _append_live_bytes(data)
        else:
            now = time.time()
            if now - last_wait_log > 1.0:
                state.tof_last_raw = (
                    f'Serial {state.tof_live_port or "auto"} odprt, '
                    f'čakam podatke; {state.tof_live_bytes} bytes'
                )
                last_wait_log = now
            time.sleep(0.02)


def _append_live_bytes(data: bytes) -> None:
    state.tof_live_bytes += len(data)
    _buffer.extend(data)

    while True:
        start = _buffer.find(SYNC)
        if start < 0:
            if len(_buffer) > 1:
                del _buffer[:-1]
            return

        if start > 0:
            del _buffer[:start]

        next_start = _buffer.find(SYNC, 2)
        if next_start < 0:
            if len(_buffer) > 65536:
                del _buffer[:-2]
            return

        raw = bytes(_buffer[:next_start])
        del _buffer[:next_start]
        state.tof_live_raw_packets += 1

        while raw and raw[-1] in (0x0A, 0x0D):
            raw = raw[:-1]

        marker = raw.find(b'Packet:')
        if marker != -1:
            raw = raw[:marker]

        try:
            packet = parse_packet(raw, raw[2])
        except Exception:
            packet = None

        if packet is None:
            state.tof_live_bad_packets += 1
            continue

        state.tof_live_packets.append(packet)
        if len(state.tof_live_packets) > MAX_LIVE_PACKETS:
            del state.tof_live_packets[:-MAX_LIVE_PACKETS]

        state.tof_packets = len(state.tof_live_packets)
        state.tof_last_seen = time.time()
        state.tof_last_raw = (
            f'{state.tof_live_bytes} bytes, '
            f'{state.tof_live_raw_packets} raw, '
            f'{state.tof_live_bad_packets} bad'
        )
        state.stm_connected = True
