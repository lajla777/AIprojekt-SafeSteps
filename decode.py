import struct
import serial
import serial.tools.list_ports
import sys
import numpy as np
import time

SERIAL_PORT = "COM7"  # change if needed (Device manager)
BAUD_RATE   = 115200
RECORD_MODE = True    # True = serial stream se sproti snema v .bin

SYNC = b'\xFF\xFF'

CHUNK_NAMES = {
    0x01: 'gyroscope',
    0x02: 'accelerometer',
    0x03: 'magnetometer',
    0x05: 'ToF sensor'
}

CHUNK_UNITS = {
    0x01: 'mdps',
    0x02: 'mg',
    0x03: 'mGauss',
    0x05: 'mm',
}

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


def parse_packet(data, packet_counter):
    if data[0:2] != SYNC:
        print("Ni sync markerja")
        return None
    
    if len(data) < 5:
        print("Paket prekratek")
        return None
    
    packet_counter = data[2]
    
    raw_payload = data[3:]
    
    payload = unstuff_bytes(raw_payload)

    if len(payload) < 8:
        print(f"Payload prekratek: {len(payload)} B")
        return None

    received_crc = struct.unpack('<H', payload[-2:])[0]
    computed_crc = crc16_compute(payload[:-2])

    if received_crc != computed_crc:
        print(f"  [#{packet_counter}] CRC napaka: "
              f"prejeto=0x{received_crc:04X}, izračunano=0x{computed_crc:04X}")
        return None

    timestamp = struct.unpack('<I', payload[0:4])[0]
    packet_size_enc = struct.unpack('<H', payload[4:6])[0]
    packet_size = packet_size_enc + 1

    chunks_data = payload[6:-2]
    pos = 0
    chunks = {}

    while pos < len(chunks_data):
        if pos + 4 > len(chunks_data):
            print(f"Nepopoln chunk header pri pos={pos}")
            break

        chunk_id = chunks_data[pos]
        chunk_size_enc = struct.unpack('<H', chunks_data[pos+1:pos+3])[0]
        chunk_size = chunk_size_enc + 1
        reserved = chunks_data[pos+3]
        chunk_data = chunks_data[pos+4:pos+4+chunk_size]

        samples = []
        if chunk_id == 0x05: 
            for i in range(0, chunk_size, 2):
                if i + 2 > len(chunk_data):
                    break
                distance = struct.unpack('<H', chunk_data[i:i+2])[0]
                samples.append((distance,))
        else:
            for i in range(0, chunk_size, 6):
                if i + 6 > len(chunk_data):
                    break
                x, y, z = struct.unpack('<hhh', chunk_data[i:i+6])
                samples.append((x, y, z))

        name = CHUNK_NAMES.get(chunk_id, f'unknown_0x{chunk_id:02X}')
        unit = CHUNK_UNITS.get(chunk_id, '?')
        chunks[chunk_id] = {'name': name, 'unit': unit, 'samples': samples}
        pos += 4 + chunk_size

    return {
        'packet_counter': packet_counter,
        'timestamp_ms':   timestamp,
        'packet_size':    packet_size,
        'chunks':         chunks,
    }


def decode_file(file_path: str) -> list:
    """Prebere .bin datoteko in vrne seznam parsiranih paketov."""

    with open(file_path, 'rb') as f:
        data = f.read()

    print(f"Velikost datoteke: {len(data)} bajtov")

    packets = []
    i = 0
    raw_count = 0

    while i < len(data) - 1:
        if data[i:i+2] != SYNC:
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

    print(f"Najdenih paketov:             {raw_count}")
    print(f"Uspešno parsiranih paketov:   {len(packets)}")
    print(f"Zavrnjenih paketov (CRC/err): {raw_count - len(packets)}") 
    return packets


def save_to_txt(packets: list, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*50}\n")
        f.write(f"Skupaj paketov: {len(packets)}\n")
        f.write(f"{'='*50}\n")

        prev_counter = None
        for pkt in packets:
            c = pkt['packet_counter']

            if prev_counter is not None:
                expected = (prev_counter + 1) % 254
                if c != expected:
                    f.write(f"\nIZGUBLJEN PAKET! Pričakovan #{expected}, prejeto #{c}\n")

            f.write(f"\n[#{c:3d}] ts={pkt['timestamp_ms']:8d} ms  size={pkt['packet_size']}\n")

            for chunk_id, chunk_data in pkt['chunks'].items():
                n = len(chunk_data['samples'])
                f.write(f"  {chunk_data['name']:15s} ({n} vzorcev, {chunk_data['unit']}):\n")
                for s in chunk_data['samples']:
                    if len(s) ==1:
                        f.write(f"distance={s[0]:7d} {chunk_data['unit']}\n")
                    else:
                        f.write(f"x={s[0]:7d}  y={s[1]:7d}  z={s[2]:7d}\n")

            prev_counter = c

    print(f"Shranjeno v: {output_path}")


def save_to_npz(packets: list, output_path: str):
    gyro_data, gyro_ts = [], []
    acc_data,  acc_ts  = [], []
    mag_data,  mag_ts  = [], []
    tof_data,  tof_ts  = [], []

    for pkt in packets:
        ts = pkt['timestamp_ms'] 

        if 0x01 in pkt['chunks']:
            for s in pkt['chunks'][0x01]['samples']:
                gyro_data.append(s)
                gyro_ts.append(ts)

        if 0x02 in pkt['chunks']:
            for s in pkt['chunks'][0x02]['samples']:
                acc_data.append(s)
                acc_ts.append(ts)

        if 0x03 in pkt['chunks']:
            for s in pkt['chunks'][0x03]['samples']:
                mag_data.append(s)
                mag_ts.append(ts)

        if 0x05 in pkt['chunks']:
            for s in pkt['chunks'][0x05]['samples']:
                tof_data.append(s[0])
                tof_ts.append(ts)

    np.savez(output_path,
        y_gyro = np.array(gyro_data, dtype=np.int16),
        t_gyro = np.array(gyro_ts,   dtype=np.uint32),
        y_acc  = np.array(acc_data,  dtype=np.int16),
        t_acc  = np.array(acc_ts,    dtype=np.uint32),
        y_mag  = np.array(mag_data,  dtype=np.int16),
        t_mag  = np.array(mag_ts,    dtype=np.uint32),
        y_tof  = np.array(tof_data,  dtype=np.uint16),
        t_tof  = np.array(tof_ts,    dtype=np.uint32)
    )
    print(f"Shranjeno v: {output_path}")
    print(f"  Žiroskop:      {len(gyro_data)} vzorcev")
    print(f"  Akcelerometer: {len(acc_data)} vzorcev")
    print(f"  Magnetometer:  {len(mag_data)} vzorcev")
    print(f"  ToF:           {len(tof_data)} vzorcev")

def stream_serial():
    """Bere pakete v živo s serijske povezave. Ctrl+C za izhod."""
    
    print(f"Opening {SERIAL_PORT} @ {BAUD_RATE} baud ...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open port: {e}")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device} — {p.description}")
        sys.exit(1)

    buf = bytearray()
    last_counter = None

    filename = f"stream_{int(time.time())}.bin" if RECORD_MODE else None
    f = open(filename, "wb") if RECORD_MODE else None
    if filename:
        print(f"Recording to {filename} ...")

    try:
        while True:
            chunk = ser.read(256)
            if chunk:
                buf.extend(chunk)

            s1 = buf.find(SYNC)
            if s1 == -1:
                buf.clear()
                continue
            if s1 > 0:
                buf = buf[s1:]

            s2 = buf.find(SYNC, 2)
            if s2 == -1:
                if len(buf) > 4096:
                    buf.clear()
                continue

            raw = bytearray(buf[:s2])
            buf = buf[s2:]

            while raw and raw[-1] in (0x0a, 0x0d):
                raw = raw[:-1]

            p_idx = raw.find(b'Packet:')
            if p_idx != -1:
                raw = raw[:p_idx]

            pkt = parse_packet(bytes(raw))
            if pkt is None:
                print(f"full raw: {raw.hex()}")
                continue

            # Zapiši v datoteko
            if f:
                f.write(bytes(raw))

            cnt = pkt['packet_counter']
            ts  = pkt['timestamp_ms']

            lost = ""
            if last_counter is not None:
                expected = (last_counter + 1) % 254
                if cnt != expected:
                    lost = f"  *** {(cnt - last_counter - 1) % 254} LOST ***"
            last_counter = cnt

            print(f"\n[#{cnt:3d}] t={ts} ms{lost}")
            for chunk in pkt['chunks'].values():
                name = chunk['name']
                for i, s in enumerate(chunk['samples']):
                    if len(s) == 1:
                        print(f"  {name}[{i}] distance={s[0]} {chunk['unit']}")
                    else:
                        print(f"  {name}[{i}] x={s[0]}  y={s[1]}  z={s[2]}")

    except KeyboardInterrupt:
        print("\nStopped.")
        if f:
            print(f"Saved: {filename}")
    finally:
        if f:
            f.close()
        ser.close()


if __name__ == "__main__":
    # Način FILE: python decode.py mojadat.bin
    # Način SERIAL: python decode.py serial
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "serial":
            stream_serial()
        else:
            packets = decode_file(arg)
            if packets:
                base = arg.rsplit('.', 1)[0]
                save_to_txt(packets, base + '_output.txt')
                save_to_npz(packets, base + '_output.npz')
            else:
                print("Ni bilo uspešno parsiranih paketov.")
    else:
        # Privzeto: file način s test.bin
        packets = decode_file('test.bin')
        if packets:
            save_to_txt(packets, 'output.txt')
            save_to_npz(packets, 'output.npz')
        else:
            print("Ni bilo uspešno parsiranih paketov.")