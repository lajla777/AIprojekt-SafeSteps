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

def sestavi_podatke(seznam_paketov):
    """
    paket_seznam = []
    st_vz_seznam = []
    matrike = []

    for i, paket in enumerate(seznam_paketov):
        st_vzorcev = paket.data.shape[0]
        st_vz_seznam.append(st_vzorcev)
        matrike.append(paket.data)

        if i<len(seznam_paketov)-1:
            dt = seznam_paketov[i+1].ts - paket.ts
            paket_seznam.append((paket.id, st_vzorcev, dt))

    paket_avg = np.mean(st_vz_seznam)
    st_vz_avg = np.mean(st_vz_seznam)
    fvz = st_vz_avg /paket_avg

    signal = np.vstack(matrike)

    return fvz, signal
    """
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

