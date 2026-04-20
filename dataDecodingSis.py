import struct
import numpy as np

GIRO_ID = 1
ACC_ID = 2
MAG_ID = 3
TOF_ID = 5

class Paket:
    def __init__(self, id, ts, data):
        self.id = id
        self.ts = ts
        self.data = data

def crc16_update(crc, data):
    crc ^= data
    for i in range(8):
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

def unstuff_bytes(data):
    unstuffed = bytearray()
    i = 0
    while i < len(data):
        if data[i] == 0xFE:
            i += 1
            if i >= len(data):
                break
            unstuffed.append(0xFE ^ data[i])
        else:
            unstuffed.append(data[i])
        i += 1
    return bytes(unstuffed)

def parse_packet(data):
    if data[0:2] != b'\xFF\xFF':
        return None

    raw = data[3:]
    payload = unstuff_bytes(raw)

    if len(payload) < 8:
        return None

    received_crc = struct.unpack('<H', payload[-2:])[0]
    computed_crc = crc16_compute(payload[:-2])

    if received_crc != computed_crc:
        return None

    timestamp = struct.unpack('<I', payload[0:4])[0]
    chunks_data = payload[6:-2]

    pos = 0
    chunks = {}

    while pos < len(chunks_data):
        chunk_id = chunks_data[pos]
        size = struct.unpack('<H', chunks_data[pos+1:pos+3])[0] + 1
        chunk_data = chunks_data[pos+4:pos+4+size]

        samples = []
        if chunk_id in [0x01, 0x02, 0x03]:
            for i in range(0, size, 6):
                if i+6 <= len(chunk_data):
                    x, y, z = struct.unpack('<hhh', chunk_data[i:i+6])
                    samples.append((x, y, z))
        elif chunk_id in [0x05]:
            for i in range(0, len(chunk_data), 2):
                if i+2 <= len(chunk_data):
                    distance = struct.unpack('<H', chunk_data[i:i+2])[0]
                    samples.append(distance)

        chunks[chunk_id] = samples
        pos += 4 + size

    return {'timestamp': timestamp, 'chunks': chunks}

def dekodiraj_bin(bin_datoteka):
    with open(bin_datoteka, "rb") as f:
        data = f.read()

    sync = b'\xFF\xFF'
    positions = []

    for i in range(len(data)-1):
        if data[i:i+2] == sync:
            positions.append(i)

    raw_packets = []
    for i in range(len(positions)):
        start = positions[i]
        end = positions[i+1] if i+1 < len(positions) else len(data)
        p = parse_packet(data[start:end])
        if p:
            raw_packets.append(p)

    seznam_paketov = []

    for p in raw_packets:
        ts = p['timestamp'] / 1000.0  #iz ms v s

        for chunk_id, samples in p['chunks'].items():
            if chunk_id in [GIRO_ID, ACC_ID, MAG_ID]:
                data_bytes = np.array(samples, dtype=np.int16).tobytes()
            elif chunk_id == TOF_ID:
                data_bytes = np.array(samples, dtype=np.uint16).tobytes()
            seznam_paketov.append(Paket(chunk_id, ts, data_bytes))

    return seznam_paketov
