import struct
import numpy as np

class Paket:
    def __init__(self, id: int, ts: float, data:np.ndarray):
        self.id = id
        self.ts = ts
        self.data = np.array(data)

    def read(self):
        return self.data
    
def crc16_update(crc, data):
    crc ^= data
    for i in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ 0xA001
        else:
            crc = crc >> 1
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

def parse_paket(data):
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

    pos=0
    chunks={}

    while pos < len(chunks_data):
        chunk_id = chunks_data[pos]
        size = struct.unpack('<H', chunks_data[pos+1:pos+3])[0] + 1
        chunk_data = chunks_data[pos+4:pos+4+size]

        samples = []
        for i in range(0, size, 6):
            if i+6 <= len(chunk_data):
                x, y, z = struct.unpack('<hhh', chunk_data[i:i+6])
                samples.append((x, y, z))

        chunks[chunk_id] = samples
        pos += 4 + size

    return {'timestamp': timestamp, 'chunks': chunks}

def sestavi_podatke(seznam_paketov, wanted_id):
    vsi = []
    T = []
    N = []

    filtrirani = [p for p in seznam_paketov if p.id == wanted_id]

    for i, p in enumerate(filtrirani):
        data = np.frombuffer(p.data, dtype=np.int16).reshape(-1, 3)
        vsi.append(data)

        N.append(data.shape[0])

        if i > 0:
            T.append(p.ts - filtrirani[i-1].ts)

    signal = np.vstack(vsi)
    Fvz = np.mean(N) / np.mean(T)

    return Fvz, signal
if __name__ == "__main__":
    with open('data.bin', 'rb') as f:
        data = f.read()
    
    #najdi vse pakete v podatkih
    sync = b'\xFF\xFF'
    positions = []
    
    for i in range(len(data)-1):
        if data[i:i+2] == sync:
            positions.append(i)

    #tu dodaj parsing

