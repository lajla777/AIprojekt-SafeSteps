import struct
import numpy as np
import matplotlib.pyplot as plt

GIRO_ID = 1
ACC_ID = 2
MAG_ID = 3
TOF_ID = 4

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

def sestavi_podatke(seznam_paketov, tofSenzor=False):
    vsi = [] #matrike podatkov iz vsakega paketa
    T = [] #casovnerazlike med paketi
    N = [] #stevilo vzorcev v vsakem paketu

    for i, p in enumerate(seznam_paketov):
        if p.id in [1, 2, 3]:
            #bytes_per_sample = 2
            #dtype = np.int16
            data = np.frombuffer(p.data, dtype=np.int16)
        elif tofSenzor:
            data = np.frombuffer(p.data, dtype=np.uint16)
        else: 
            #dtype = np.uint8
            #bytes_per_sample = 1
            data = np.frombuffer(p.data, dtype=np.uint8)

        #data = np.frombuffer(p.data, dtype=dtype)

        Nvz = len(data) if tofSenzor else len(data) // 3
        N.append(Nvz)

        if not tofSenzor:
            data = data.reshape(-1, 3)
        vsi.append(data)

        if i > 0:
            T.append(p.ts - seznam_paketov[i-1].ts)

    signal = np.vstack(vsi)

    if len(T) > 0:
        Fvz = np.mean(N) / np.mean(T)
    else:
        Fvz = 0

    return Fvz, signal

def prikazi_signal(signal, naslov=None, startInd=None, endInd=None):
    if startInd is None:
        startInd = 0
    if endInd is None:
        endInd = signal.shape[0]

    sig = signal[startInd:endInd]
    x = np.arange(startInd, endInd)

    plt.figure(figsize=(13, 6))

    plt.plot(x, sig[:,0], label="X", color='#B063F8')
    plt.plot(x, sig[:,1], label="Y", color="#FC78F3")
    plt.plot(x, sig[:,2], label="Z", color='#A80354')

    plt.title(naslov)
    if "Žiroskop" in naslov:
        plt.ylabel("rotacija (°/s)")
    elif "Akcelometer" in naslov:
        plt.ylabel("pospešek (G)")
    elif "Magnetometer" in naslov:
        plt.ylabel("magnetno polje (Gauss)")
    plt.xlabel("vzorec")
    plt.legend()
    plt.grid()

    plt.show()

def signali_skupaj(bin_datoteka):
    vsiPaketi = dekodiraj_bin(bin_datoteka)

    paketiGiro = [p for p in vsiPaketi if p.id == GIRO_ID]
    paketiAcc = [p for p in vsiPaketi if p.id == ACC_ID]
    paketiMag = [p for p in vsiPaketi if p.id == MAG_ID]

    FvzGiro, signalGiro = sestavi_podatke(paketiGiro)
    FvzAcc, signalAcc = sestavi_podatke(paketiAcc)
    FvzMag, signalMag = sestavi_podatke(paketiMag)

    signalGiro = signalGiro * 8.75e-3
    signalAcc = signalAcc * 6.125e-5
    signalMag = signalMag * 1.5e-3

    #casovne osi za vsak signal
    tGiro = np.arange(len(signalGiro)) / FvzGiro
    tAcc = np.arange(len(signalAcc)) / FvzAcc
    tMag = np.arange(len(signalMag)) / FvzMag

    plt.figure(figsize=(11, 7))
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

if __name__ == "__main__":
    print("VIZUALIZACIJA PODATKOV")
    vsiPaketi = dekodiraj_bin("log10.bin")

    while True:
        print("\nKateri signal želiš prikazati?")
        print("- vsi signali skupaj (testni): 1")
        print("- vsi signali skupaj (moji): 2")
        print("- žiroskop: 3")
        print("- akcelometer: 4")
        print("- magnetometer: 5")

        izbira = input("\nIzberi možnost (1-5): ")

        if izbira == "1":
            signali_skupaj("LOG011.BIN")  
        elif izbira == "2":
            signali_skupaj("log10.bin")
            
        elif izbira == "3" or izbira == "4" or izbira == "5":

            if izbira == "3":
                paketiGiro = [p for p in vsiPaketi if p.id == GIRO_ID]
                FvzGiro, signalGiro = sestavi_podatke(paketiGiro)
                signalGiro = signalGiro * 8.75e-3
                print(f"Fvz iroskopa= {FvzGiro:.2f} Hz")

                prikazi_signal(signalGiro, f"Žiroskop (Fvz={FvzGiro:.2f} Hz)")
                prikazi_signal(signalGiro,
                            "Žiroskop - interval",
                            1,
                            int(FvzGiro * 3)+1)
                #x os = rotacija naprej/nazaj
                #y os = rotacija levo/desno
                #z os = rotacija okoli svoje osi

            elif izbira == "4":
                paketiAcc = [p for p in vsiPaketi if p.id == ACC_ID]
                FvzAcc, signalAcc = sestavi_podatke(paketiAcc)
                signalAcc = signalAcc * 6.125e-5 
                print(f"Fvz akcelometera = {FvzAcc:.2f} Hz")

                prikazi_signal(signalAcc, f"Akcelometer (Fvz={FvzAcc:.2f} Hz)")
                prikazi_signal(signalAcc,
                            "Akcelometer - interval",
                            1,
                            int(FvzAcc * 3)+1)
                #x os = pospešek naprej/nazaj
                #y os = pospešek levo/desno
                #z os = pospešek gor/dol(gravitacija)

            elif izbira == "5":
                paketiMag = [p for p in vsiPaketi if p.id == MAG_ID]
                FvzMag, signalMag = sestavi_podatke(paketiMag)
                signalMag = signalMag * 1.5e-3
                print(f"Fvz magnetometra = {FvzMag:.2f} Hz")    

                prikazi_signal(signalMag, f"Magnetometer (Fvz={FvzMag:.2f} Hz)")
                prikazi_signal(signalMag,   
                            "Magnetometer - interval",
                            1,
                            int(FvzMag * 3)+1)
                #x os = magnetno polje naprej/nazaj
                #y os = magnetno polje levo/desno
                #z os = magnetno polje gor/dol
