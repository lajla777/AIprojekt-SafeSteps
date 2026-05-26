#!/usr/bin/env python3
import os
import socket
import struct
import threading
import time
from datetime import datetime

import numpy as np
import serial
import serial.tools.list_ports

HOST = "127.0.0.1"
PORT = 5000

STM32_VID = "0483"
STM32_PID = "5740"
BAUDRATE = 115200
SERIAL_TIMEOUT = 2
SCAN_INTERVAL = 2
WORK_DIR = os.getcwd()
STREAM_CAPTURE_SECONDS = 5

SYNC = b"\xFF\xFF"

GYRO_ID = 0x01
ACC_ID = 0x02
MAG_ID = 0x03
TOF_ID = 0x05

CHUNK_NAMES = {
    GYRO_ID: "gyroscope",
    ACC_ID: "accelerometer",
    MAG_ID: "magnetometer",
    TOF_ID: "ToF sensor",
}

CHUNK_UNITS = {
    GYRO_ID: "mdps",
    ACC_ID: "mg",
    MAG_ID: "mGauss",
    TOF_ID: "mm",
}

class ServiceState:
    def __init__(self):
        self.lock = threading.Lock()
        self.serial_lock = threading.Lock()
        self.serial_conn = None
        self.device_path = None
        self.clients = []
        self.running = True

state = ServiceState()

def broadcast(message):
    with state.lock:
        clients = list(state.clients)

    for conn in clients:
        try:
            conn.sendall(message.encode("utf-8", errors="ignore"))
        except OSError:
            remove_client(conn)

def log(message):
    text = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(text, flush=True)
    broadcast(message + "\n")

def add_client(conn):
    with state.lock:
        state.clients.append(conn)

def remove_client(conn):
    with state.lock:
        if conn in state.clients:
            state.clients.remove(conn)

def find_stm32_port():
    for port in serial.tools.list_ports.comports():
        vid = f"{port.vid:04X}" if port.vid is not None else ""
        pid = f"{port.pid:04X}" if port.pid is not None else ""

        if vid == STM32_VID and pid == STM32_PID:
            return port.device

        hwid = (port.hwid or "").upper()
        if f"VID:PID={STM32_VID}:{STM32_PID}" in hwid:
            return port.device

    return None

def is_stm32_connected():
    with state.lock:
        return state.serial_conn is not None and state.device_path is not None

def get_device_path():
    with state.lock:
        return state.device_path

def close_serial_connection():
    with state.lock:
        conn = state.serial_conn
        state.serial_conn = None
        state.device_path = None

    if conn is not None:
        try:
            conn.close()
        except OSError:
            pass

def device_watcher():
    while state.running:
        try:
            detected_port = find_stm32_port()

            with state.lock:
                current_port = state.device_path
                current_conn = state.serial_conn

            if detected_port is None:
                if current_conn is not None:
                    close_serial_connection()
                    log("STM32 has disconnected")

            elif current_conn is None or detected_port != current_port:
                close_serial_connection()
                try:
                    ser = serial.Serial(detected_port, BAUDRATE, timeout=SERIAL_TIMEOUT)
                    time.sleep(1)
                    with state.lock:
                        state.serial_conn = ser
                        state.device_path = detected_port
                    log(f"STM32 detected at {detected_port}")
                except serial.SerialException as e:
                    close_serial_connection()
                    log(f"FAIL: could not open STM32 at {detected_port}: {e}")

        except Exception as e:
            log(f"FAIL: device watcher error: {e}")

        time.sleep(SCAN_INTERVAL)

#dekodiranje (zdej tkko na dev branch)
def crc16_update(crc, data):
    crc ^= data
    for _ in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ 0xA001
        else:
            crc >>= 1
    return crc

def crc16_compute(data):
    crc = 0xFFFF
    for byte in data:
        crc = crc16_update(crc, byte)
    return crc

def unstuff_bytes(data: bytes) -> bytes:
    unstuffed = bytearray()
    i = 0
    while i < len(data):
        if data[i] == 0xFE:
            i += 1
            if i < len(data):
                unstuffed.append(data[i] ^ 0xFE)
        else:
            unstuffed.append(data[i])
        i += 1
    return bytes(unstuffed)

def parse_packet(data: bytes, packet_counter=None):
    if len(data) < 5:
        log("Paket prekratek")
        return None

    if data[0:2] != SYNC:
        log("Ni sync markerja")
        return None

    packet_counter = data[2]
    raw_payload = data[3:]
    payload = unstuff_bytes(raw_payload)

    if len(payload) < 8:
        log(f"Payload prekratek: {len(payload)} B")
        return None

    received_crc = struct.unpack("<H", payload[-2:])[0]
    computed_crc = crc16_compute(payload[:-2])

    if received_crc != computed_crc:
        log(
            f"[#{packet_counter}] CRC napaka: "
            f"prejeto=0x{received_crc:04X}, izracunano=0x{computed_crc:04X}"
        )
        return None

    timestamp = struct.unpack("<I", payload[0:4])[0]
    packet_size_enc = struct.unpack("<H", payload[4:6])[0]
    packet_size = packet_size_enc + 1

    chunks_data = payload[6:-2]
    pos = 0
    chunks = {}

    while pos < len(chunks_data):
        if pos + 4 > len(chunks_data):
            log(f"Nepopoln chunk header pri pos={pos}")
            break

        chunk_id = chunks_data[pos]
        chunk_size_enc = struct.unpack("<H", chunks_data[pos + 1:pos + 3])[0]
        chunk_size = chunk_size_enc + 1
        chunk_data = chunks_data[pos + 4:pos + 4 + chunk_size]

        if len(chunk_data) < chunk_size:
            log(f"Nepopolni podatki chunka 0x{chunk_id:02X}")
            break

        samples = []
        if chunk_id == TOF_ID:
            for i in range(0, chunk_size, 2):
                if i + 2 > len(chunk_data):
                    break
                distance = struct.unpack("<H", chunk_data[i:i + 2])[0]
                samples.append((distance,))
        else:
            for i in range(0, chunk_size, 6):
                if i + 6 > len(chunk_data):
                    break
                x, y, z = struct.unpack("<hhh", chunk_data[i:i + 6])
                samples.append((x, y, z))

        name = CHUNK_NAMES.get(chunk_id, f"unknown_0x{chunk_id:02X}")
        unit = CHUNK_UNITS.get(chunk_id, "?")
        chunks[chunk_id] = {"name": name, "unit": unit, "samples": samples}
        pos += 4 + chunk_size

    return {
        "packet_counter": packet_counter,
        "timestamp_ms": timestamp,
        "packet_size": packet_size,
        "chunks": chunks,
    }

def decode_file(file_path: str) -> list:
    with open(file_path, "rb") as f:
        data = f.read()

    log(f"Velikost datoteke {os.path.basename(file_path)}: {len(data)} bajtov")

    packets = []
    i = 0
    raw_count = 0

    while i < len(data) - 1:
        if data[i:i + 2] != SYNC:
            i += 1
            continue

        raw_count += 1
        packet_counter = data[i + 2]

        next_sync = data.find(SYNC, i + 3)
        if next_sync == -1:
            next_sync = len(data)

        raw_packet = data[i:next_sync]
        pkt = parse_packet(raw_packet, packet_counter)
        if pkt:
            packets.append(pkt)

        i = next_sync

    log(f"Najdenih paketov: {raw_count}")
    log(f"Uspesno parsiranih paketov: {len(packets)}")
    log(f"Zavrnjenih paketov CRC/err: {raw_count - len(packets)}")
    return packets

def save_to_txt(packets: list, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 50}\n")
        f.write(f"Skupaj paketov: {len(packets)}\n")
        f.write(f"{'=' * 50}\n")

        prev_counter = None
        for pkt in packets:
            c = pkt["packet_counter"]

            if prev_counter is not None:
                expected = (prev_counter + 1) % 254
                if c != expected:
                    f.write(f"\nIZGUBLJEN PAKET! Pricakovan #{expected}, prejeto #{c}\n")

            f.write(f"\n[#{c:3d}] ts={pkt['timestamp_ms']:8d} ms  size={pkt['packet_size']}\n")

            for chunk_data in pkt["chunks"].values():
                n = len(chunk_data["samples"])
                f.write(f"  {chunk_data['name']:15s} ({n} vzorcev, {chunk_data['unit']}):\n")
                for s in chunk_data["samples"]:
                    if len(s) == 1:
                        f.write(f"distance={s[0]:7d} {chunk_data['unit']}\n")
                    else:
                        f.write(f"x={s[0]:7d}  y={s[1]:7d}  z={s[2]:7d}\n")

            prev_counter = c

    log(f"Shranjeno v: {output_path}")

def save_to_npz(packets: list, output_path: str):
    gyro_data, gyro_ts = [], []
    acc_data, acc_ts = [], []
    mag_data, mag_ts = [], []
    tof_data, tof_ts = [], []

    for pkt in packets:
        ts = pkt["timestamp_ms"]

        if GYRO_ID in pkt["chunks"]:
            for s in pkt["chunks"][GYRO_ID]["samples"]:
                gyro_data.append(s)
                gyro_ts.append(ts)

        if ACC_ID in pkt["chunks"]:
            for s in pkt["chunks"][ACC_ID]["samples"]:
                acc_data.append(s)
                acc_ts.append(ts)

        if MAG_ID in pkt["chunks"]:
            for s in pkt["chunks"][MAG_ID]["samples"]:
                mag_data.append(s)
                mag_ts.append(ts)

        if TOF_ID in pkt["chunks"]:
            for s in pkt["chunks"][TOF_ID]["samples"]:
                tof_data.append(s[0])
                tof_ts.append(ts)

    np.savez(
        output_path,
        y_gyro=np.array(gyro_data, dtype=np.int16),
        t_gyro=np.array(gyro_ts, dtype=np.uint32),
        y_acc=np.array(acc_data, dtype=np.int16),
        t_acc=np.array(acc_ts, dtype=np.uint32),
        y_mag=np.array(mag_data, dtype=np.int16),
        t_mag=np.array(mag_ts, dtype=np.uint32),
        y_tof=np.array(tof_data, dtype=np.uint16),
        t_tof=np.array(tof_ts, dtype=np.uint32),
    )

    log(f"Shranjeno v: {output_path}")
    log(f"Ziroskop: {len(gyro_data)} vzorcev")
    log(f"Akcelerometer: {len(acc_data)} vzorcev")
    log(f"Magnetometer: {len(mag_data)} vzorcev")
    log(f"ToF: {len(tof_data)} vzorcev")

def process_file(filename):
    path = os.path.join(WORK_DIR, filename)

    if not os.path.exists(path):
        return False, f"FAIL: file {filename} was not saved"

    try:
        packets = decode_file(path)
        base = os.path.splitext(filename)[0]
        txt_path = os.path.join(WORK_DIR, base + "_output.txt")
        npz_path = os.path.join(WORK_DIR, base + "_output.npz")

        if not packets:
            return False, f"FAIL: no valid packets in {filename}"

        save_to_txt(packets, txt_path)
        save_to_npz(packets, npz_path)
        return True, f"{os.path.basename(txt_path)}, {os.path.basename(npz_path)}"

    except Exception as e:
        return False, f"FAIL: processing {filename} failed: {e}"


def capture_stream_from_serial(ser, seconds=STREAM_CAPTURE_SECONDS):
    """
    Fallback način za STM32, ki ne podpira datotečnih ukazov.
    Nekaj sekund bere binarni serial stream, ga shrani v .bin in obdela.
    """
    
    filename = f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
    path = os.path.join(WORK_DIR, filename)

    end_time = time.time() + seconds
    total = 0

    log(f"Fallback stream capture started ({seconds} s)")

    with open(path, "wb") as f:
        while time.time() < end_time:
            chunk = ser.read(512)
            if chunk:
                f.write(chunk)
                total += len(chunk)

    if total == 0:
        return False, "FAIL: no data received from STM32 stream"

    log(f"Fallback stream saved {total} bytes to {filename}")

    ok, processed = process_file(filename)
    if not ok:
        return False, processed

    log(f"Processed outputs: {processed}")
    return True, filename

#komunikacija s stm
def read_line_from_serial(ser):
    line = ser.readline()
    if not line:
        return ""
    return line.decode("utf-8", errors="ignore").strip()

def receive_file_from_serial(ser, header):
    parts = header.split("|")

    if len(parts) != 3:
        return False, "FAIL: invalid FILE header from STM32"

    filename = os.path.basename(parts[1])

    try:
        size = int(parts[2])
    except ValueError:
        return False, "FAIL: invalid file size from STM32"

    path = os.path.join(WORK_DIR, filename)
    remaining = size

    with open(path, "wb") as f:
        while remaining > 0:
            chunk = ser.read(min(1024, remaining))
            if not chunk:
                return False, "FAIL: file transfer interrupted"
            f.write(chunk)
            remaining -= len(chunk)

    ok, processed = process_file(filename)
    if not ok:
        return False, processed

    log(f"Processed outputs: {processed}")
    return True, filename

def stm32_list_files(ser):
    """Pošlje LIST ukaz STM32 in vrne seznam imen datotek."""
    ser.reset_input_buffer()
    ser.write(b"LIST\n")
    ser.flush()

    files = []
    deadline = time.time() + 10
    while time.time() < deadline:
        line = read_line_from_serial(ser)
        if not line:
            continue
        if line.strip() == ">":
            break
        parts = line.split()
        if len(parts) == 2 and parts[0].upper().endswith(".BIN"):
            files.append(parts[0])

    return files


def stm32_get_file(ser, filename):
    """Pošlje GET <filename> STM32 in shrani binarno vsebino na disk."""
    ser.reset_input_buffer()
    ser.write(f"GET {filename}\n".encode("utf-8"))
    ser.flush()

    path = os.path.join(WORK_DIR, filename)
    total = 0
    deadline = time.time() + 60

    with open(path, "wb") as f:
        while time.time() < deadline:
            chunk = ser.read(512)
            if chunk:
                f.write(chunk)
                total += len(chunk)
                deadline = time.time() + 5
            else:
                if ser.in_waiting == 0 and total > 0:
                    break

    if total == 0:
        return False, f"FAIL: no data received for {filename}"

    log(f"Received {total} bytes for {filename}")
    return True, filename


def execute_stm32_command(command):
    if not is_stm32_connected():
        return "FAIL: STM32 is not connected\n"

    with state.serial_lock:
        with state.lock:
            ser = state.serial_conn

        if ser is None:
            return "FAIL: STM32 is not connected\n"

        try:
            if command == "GET_ALL":
                files = stm32_list_files(ser)
                if not files:
                    return "FAIL: no files found on STM32\n"
                for filename in files:
                    ok, result = stm32_get_file(ser, filename)
                    if not ok:
                        return result + "\n"
                    ok, processed = process_file(filename)
                    if not ok:
                        return processed + "\n"
                    log(f"Processed: {processed}")
                return "All files from STM32 are processed\n"

            elif command == "GET_LAST":
                files = stm32_list_files(ser)
                if not files:
                    return "FAIL: no files found on STM32\n"
                filename = files[-1]
                ok, result = stm32_get_file(ser, filename)
                if not ok:
                    return result + "\n"
                ok, processed = process_file(filename)
                if not ok:
                    return processed + "\n"
                log(f"Processed: {processed}")
                return "Last file from STM32 has been processed\n"

            elif command.startswith("GET_FILE|"):
                filename = command.split("|", 1)[1].strip()
                ok, result = stm32_get_file(ser, filename)
                if not ok:
                    return result + "\n"
                ok, processed = process_file(filename)
                if not ok:
                    return processed + "\n"
                log(f"Processed: {processed}")
                return f"File {filename} from STM32 has been processed\n"

            elif command == "DELETE":
                ser.reset_input_buffer()
                ser.write(b"DELETE\n")
                ser.flush()
                deadline = time.time() + 10
                while time.time() < deadline:
                    line = read_line_from_serial(ser)
                    if line.strip() == ">":
                        break
                return "All files on STM32 are deleted\n"

            else:
                return "FAIL: unknown command\n"

        except (serial.SerialException, OSError):
            close_serial_connection()
            log("STM32 has disconnected")
            return "FAIL: STM32 is not connected\n"
        except Exception as e:
            return f"FAIL: {e}\n"

def command_status():
    if is_stm32_connected():
        return "STM32 is connected\n"
    return "STM32 is not connected\n"

def handle_command(command):
    command = command.strip()

    if command == "STATUS":
        return command_status()

    if command in ["GET_LAST", "GET_ALL", "DELETE"]:
        return execute_stm32_command(command)

    if command.startswith("GET_FILE|"):
        filename = command.split("|", 1)[1].strip()
        if not filename:
            return "FAIL: missing file name\n"
        return execute_stm32_command(f"GET_FILE|{filename}")

    return f"FAIL: unknown command {command}\n"

def handle_client(conn, addr):
    add_client(conn)

    try:
        with conn:
            if is_stm32_connected():
                conn.sendall(
                    f"Connected to SPO STM32 service - STM32 detected at {get_device_path()}\n".encode()
                )
            else:
                conn.sendall(b"Connected to SPO STM32 service - No STM32 detected\n")

            buffer = ""

            while True:
                data = conn.recv(1024)
                if not data:
                    break

                buffer += data.decode("utf-8", errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    response = handle_command(line)
                    conn.sendall(response.encode("utf-8", errors="ignore"))

                # Podpora za PowerShell ročni test brez \n.
                if buffer.strip() and not buffer.endswith("|"):
                    command = buffer.strip()
                    buffer = ""
                    response = handle_command(command)
                    conn.sendall(response.encode("utf-8", errors="ignore"))

    except OSError:
        pass
    finally:
        remove_client(conn)

def main():
    watcher_thread = threading.Thread(target=device_watcher, daemon=True)
    watcher_thread.start()

    log(f"SPO STM32 service started on {HOST}:{PORT}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()

        while True:
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            thread.start()


if __name__ == "__main__":
    main()
