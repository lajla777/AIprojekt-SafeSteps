import struct
import numpy as np
import matplotlib.pyplot as plt

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

def prikazi_signal(signal, naslov="", startInd=None, endInd=None):
    if startInd is None:
        startInd = 0
    if endInd is None:
        endInd = signal.shape[0]

    sig = signal[startInd:endInd]

    plt.figure(figsize=(10, 6))

    plt.plot(sig[:,0], label="X")
    plt.plot(sig[:,1], label="Y")
    plt.plot(sig[:,2], label="Z")

    plt.title(naslov)
    plt.xlabel("vzorec")
    plt.ylabel("vrednost")
    plt.legend()
    plt.grid()

    plt.show()

def signali_skupaj(bin_datoteka):
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
        ts = p['timestamp'] / 1000.0
        for chunk_id, samples in p['chunks'].items():
            data_bytes = np.array(samples, dtype=np.int16).tobytes()
            seznam_paketov.append(Paket(chunk_id, ts, data_bytes))


    FvzGiro, signalGiro   = sestavi_podatke(seznam_paketov, 1)
    FvzAcc, signalAcc  = sestavi_podatke(seznam_paketov, 2)
    FvzMag, signalMag    = sestavi_podatke(seznam_paketov, 3)

    signalGiro  = signalGiro  * 8.75e-3
    signalAcc = signalAcc * 6.125e-5
    signalMag   = signalMag   * 1.5e-3

    #casovne osi za vsak signal
    tGiro  = np.arange(len(signalGiro))  / FvzGiro
    tAcc = np.arange(len(signalAcc)) / FvzAcc
    tMag   = np.arange(len(signalMag))   / FvzMag

    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Sensor data from {bin_datoteka}", fontsize=16)

    plt.subplot(3,1,1)
    plt.plot(tGiro, signalGiro[:,0], label="x")
    plt.plot(tGiro, signalGiro[:,1], label="y")
    plt.plot(tGiro, signalGiro[:,2], label="z")
    plt.title(f"Gyroscope (Fvz={FvzGiro:.1f} Hz, resolution 8.75e-3 °/s)")
    plt.ylabel("rotational speed (°/s)")
    plt.legend()
    plt.grid()

    plt.subplot(3,1,2)
    plt.plot(tAcc, signalAcc[:,0], label="x")
    plt.plot(tAcc, signalAcc[:,1], label="y")
    plt.plot(tAcc, signalAcc[:,2], label="z")
    plt.title(f"Accelerometer (Fvz={FvzAcc:.1f} Hz, resolution 6.125e-5 g)")
    plt.ylabel("acceleration (G)")
    plt.legend()
    plt.grid()

    plt.subplot(3,1,3)
    plt.plot(tMag, signalMag[:,0], label="x")
    plt.plot(tMag, signalMag[:,1], label="y")
    plt.plot(tMag, signalMag[:,2], label="z")
    plt.title(f"Magnetometer (Fvz={FvzMag:.1f} Hz, resolution 1.5e-3 Gauss)")
    plt.xlabel("time (s)")
    plt.ylabel("magnetic field (Gauss)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def main():
    signali_skupaj("LOG011.BIN")  
    signali_skupaj("dataTestBIN.bin")
    
    with open("dataTestBIN.bin", "rb") as f:
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
            data_bytes = np.array(samples, dtype=np.int16).tobytes()
            seznam_paketov.append(Paket(chunk_id, ts, data_bytes))

    FvzGiro, signalGiro = sestavi_podatke(seznam_paketov, 1)
    signalGiro = signalGiro * 8.75e-3
    print(f"Fvz žiroskopa= {FvzGiro:.2f} Hz")

    prikazi_signal(signalGiro, f"Žiroskop - cel signal (Fvz={FvzGiro:.2f} Hz)")
    prikazi_signal(signalGiro,
                   "Žiroskop - interval",
                   0,
                   int(FvzGiro * 2))
    
    FvzAcc, signalAcc = sestavi_podatke(seznam_paketov, 2)
    signalAcc = signalAcc * 6.125e-5 
    print(f"Fvz akcelometera = {FvzAcc:.2f} Hz")

    prikazi_signal(signalAcc, f"Akcelometer - cel signal (Fvz={FvzAcc:.2f} Hz)")
    prikazi_signal(signalAcc,
                   "Akcelometer - interval",
                   0,
                   int(FvzAcc * 2))
    
    FvzMag, signalMag = sestavi_podatke(seznam_paketov, 3)
    signalMag = signalMag * 1.5e-3
    print(f"Fvz magnetometra = {FvzMag:.2f} Hz")    

    prikazi_signal(signalMag, f"Magnetometer - cel signal (Fvz={FvzMag:.2f} Hz)")
    prikazi_signal(signalMag,   
                   "Magnetometer - interval",
                   0,
                   int(FvzMag * 2))

if __name__ == "__main__":
    main()
