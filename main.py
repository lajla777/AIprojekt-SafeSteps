import struct
import numpy as np

SYNC = b'\xFF\xFF'

def destuff(data: bytes) -> bytes:
    result = bytearray()
    i = 0
    while i < len(data):
        if data[i] == 0xFE:
            i += 1
            if i < len(data):
                result.append(data[i] ^ 0xFE)
        else:
            result.append(data[i])
        i += 1
    return bytes(result)

def crc16_update(crc, data):
    crc ^= data
    for _ in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ 0xA001
        else:
            crc = crc >> 1
    return crc

def crc16_compute(data, length):
    crc = 0xFFFF
    for i in range(length):
        crc = crc16_update(crc, data[i])
    return crc

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

def parse_packet(data, packet_counter):
    if data[0:2] != b'\xFF\xFF':
        print(f"Ni sync markerja")
        return None

    payload = destuff(data[3:]) 

    if len(payload) < 8:
        print(f"Payload prekratek: {len(payload)} B")
        return None

    timestamp = struct.unpack('<I', payload[0:4])[0]
    packet_size_enc = struct.unpack('<H', payload[4:6])[0]
    packet_size = packet_size_enc + 1

    received_crc = struct.unpack('<H', payload[-2:])[0]
    computed_crc = crc16_compute(payload[:-2], len(payload) - 2)
    crc_ok = received_crc == computed_crc
    if not crc_ok:
        print(f"CRC napaka: prejeto=0x{received_crc:04X}, izračunano=0x{computed_crc:04X}")

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
        'crc_ok':         crc_ok,
    }


def decodeSpo(file_path: str) -> list:
    with open(file_path, 'rb') as f:
        data = f.read()

    packets = []
    i = 0

    while i < len(data) - 1:
        if data[i:i+2] != SYNC:
            i += 1
            continue

        packet_counter = data[i + 2]

        next_sync = data.find(SYNC, i + 3)
        if next_sync == -1:
            next_sync = len(data)

        raw_packet = data[i:next_sync]

        pkt = parse_packet(raw_packet, packet_counter)
        if pkt:
            packets.append(pkt)

        i = next_sync

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

            crc_str = "OK" if pkt['crc_ok'] else "NAPAKA"
            f.write(f"\n[#{c:3d}] ts={pkt['timestamp_ms']:8d} ms  size={pkt['packet_size']}  CRC {crc_str}\n")

            for cid, chunk in pkt['chunks'].items():
                n = len(chunk['samples'])
                f.write(f"  {chunk['name']:15s} ({n} vzorcev, {chunk['unit']}):\n")
                for s in chunk['samples']:
                    if len(s) ==1:
                        f.write(f"distance={s[0]:7d} {chunk['unit']}\n")
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


if __name__ == "__main__":
    fn = 'test.bin'
    packets = decodeSpo(fn)
    save_to_txt(packets, 'output.txt')
    save_to_npz(packets, 'output.npz')