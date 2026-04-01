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
    
TIP_ZLOGA = {
    1: 2, #1=ziroskop 2=pospeskomeer in 3=magnetometer
    2: 2,
    3: 2
}

def sestavi_podatke(seznam_paketov):
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

if __name__ == "__main__":
    pass
    
