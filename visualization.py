import numpy as np
from dataclasses import dataclass
from decode import decode_file
import matplotlib.pyplot as plt

@dataclass
class Paket:
    id:   int
    ts:   float
    data: np.ndarray

def v_pakete(packets) -> list:
    seznam = []
    for pkt in packets:
        ts = pkt['timestamp_ms'] / 1000.0
        for chunk_id, chunk in pkt['chunks'].items():
            raw = bytearray()
            for s in chunk['samples']:
                for vrednost in s:
                    if chunk_id in [0x01, 0x02, 0x03]: 
                        raw += np.int16(vrednost).tobytes()
                    elif chunk_id == 0x05:  
                        raw += np.uint16(vrednost).tobytes()
            seznam.append(Paket(
                id=chunk_id,
                ts=ts,
                data=np.frombuffer(raw, dtype=np.uint8)
            ))
    return seznam

BYTES_PER_SAMPLE = {
    0x01: 2,
    0x02: 2,
    0x03: 2,
    0x05: 2,
}
KOORDINATE = {
    0x01: 3,
    0x02: 3,
    0x03: 3,
    0x05: 1,
}

def sestavi_podatke(seznam_paketov: list):
    if not seznam_paketov:
        raise ValueError("Seznam paketov je prazen")

    chunk_id = seznam_paketov[0].id
    bpk = BYTES_PER_SAMPLE.get(chunk_id, 2)
    koordinate = KOORDINATE.get(chunk_id, 3)
    bytes_na_vzorec = bpk * koordinate

    T_paketi = []
    Nvz_paketi = []

    for i in range(1, len(seznam_paketov)):
        dt = seznam_paketov[i].ts - seznam_paketov[i-1].ts
        if dt > 0:
            T_paketi.append(dt)
        n_bytes = len(seznam_paketov[i-1].data)
        Nvz_paketi.append(n_bytes // bytes_na_vzorec)

    Nvz_paketi.append(len(seznam_paketov[-1].data) // bytes_na_vzorec)

    Tpaket_avg = np.mean(T_paketi)
    Nvz_avg    = np.mean(Nvz_paketi)
    Fvz = Nvz_avg / Tpaket_avg

    vsi_vzorci = []
    for paket in seznam_paketov:
        n_vzorcev = len(paket.data) // bytes_na_vzorec
        for i in range(n_vzorcev):
            offset = i * bytes_na_vzorec
            vzorec = []
            for k in range(koordinate):
                b = paket.data[offset + k*bpk : offset + k*bpk + bpk]
                is_signed = chunk_id in [0x01, 0x02, 0x03]
                vrednost = int.from_bytes(b, byteorder='little', signed=is_signed)
                if chunk_id == 0x05:
                    vzorec.append(vrednost if vrednost != 0xFFFF else np.nan)
                else:
                    vzorec.append(vrednost)
            vsi_vzorci.append(vzorec)

    dtype = np.float32 if chunk_id == 0x05 else np.int16
    matrika = np.array(vsi_vzorci, dtype=dtype)
    return Fvz, matrika


def prikazi_signal(signal: np.ndarray,
                   naslov: str = None,
                   startInd: int = None,
                   endInd: int = None,
                   Fvz: float = None,
                   enota: str = 'vrednost', 
                   ax=None):
    
    if startInd is not None or endInd is not None:
        signal = signal[startInd:endInd]

    N, K = signal.shape
    koordinate = ['X', 'Y', 'Z'] if K <= 3 else [str(i) for i in range(K)]

    if Fvz is not None and Fvz > 0:
        offset = (startInd/Fvz) if startInd is not None else 0
        x_os = (np.arange(N) / Fvz) + offset
        x_label = 'Čas [s]'
    else:
        x_os = np.arange(N)
        x_label = 'Vzorec'

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 6))

    barve = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for k in range(K):
        label = koordinate[k] if K > 1 else 'signal'
        ax.plot(x_os, signal[:, k], label=label, color=barve[k % len(barve)])

    title = naslov if naslov else ''
    if Fvz is not None and Fvz > 0:
        title += f', sampling rate: {Fvz:.1f} Hz'
    if title:
        ax.set_title(title, fontsize=10)

    ax.set_ylabel(f'{enota}', fontsize=9)
    if K > 1:
        ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.4)

    if standalone:
        ax.set_xlabel(x_label, fontsize=11)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    for i in range(1, 25):
        file = 'Zajemanje podatkov/Oseba na prostem/LOG%03i.BIN' % i
        packets = decode_file(file)
        seznam  = v_pakete(packets)

        gyro_paketi  = [p for p in seznam if p.id == 0x01]
        accel_paketi = [p for p in seznam if p.id == 0x02]
        mag_paketi   = [p for p in seznam if p.id == 0x03]
        tof_paketi   = [p for p in seznam if p.id == 0x05]

        Fvz_gyro,  gyro_mat  = sestavi_podatke(gyro_paketi)
        Fvz_accel, accel_mat = sestavi_podatke(accel_paketi)
        Fvz_mag,   mag_mat   = sestavi_podatke(mag_paketi)
        Fvz_tof,   tof_mat   = sestavi_podatke(tof_paketi)

        accel_mat_g = accel_mat * 6.125e-5
        gyro_mat_dps = gyro_mat * 8.75e-3
        mag_mat_Gauss = mag_mat * 1.5e-3


        print(f"Gyroscope:     Fvz={Fvz_gyro:.1f} Hz,  matrika={gyro_mat.shape}")
        print(f"Accelerometer: Fvz={Fvz_accel:.1f} Hz, matrika={accel_mat.shape}")
        print(f"Magnetometer:  Fvz={Fvz_mag:.1f} Hz,  matrika={mag_mat.shape}")
        print(f"ToF sensor:    Fvz={Fvz_tof:.1f} Hz,  matrika={tof_mat.shape}")

        fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(f'Vsi senzorji — {file}', fontsize=15, y=0.98)

        prikazi_signal(gyro_mat_dps, naslov=f'Gyroscope\n resolution 8.75e-3 °/s',
                Fvz=Fvz_gyro, enota='rotational speed (dps)', ax=axes[0])

        prikazi_signal(accel_mat_g, naslov=f'Accelerometer\nresolution 6.125e-5 g-force',
                    Fvz=Fvz_accel, enota='acceleration (G-force)', ax=axes[1])

        prikazi_signal(mag_mat_Gauss,naslov=f'Magnetometer\nresolution 1.5e-3 Gauss',
                Fvz=Fvz_mag, enota='magnetic field (Gauss)', ax=axes[2])
        
        prikazi_signal(tof_mat, naslov=f'Time-of-Flight\nresolution 1 mm',
                Fvz=Fvz_tof, enota='distance (mm)', ax=axes[3])

        axes[-1].set_xlabel('Čas [s]', fontsize=10)
        fig.subplots_adjust(hspace=0.6, top=0.90, bottom=0.06) 
        #plt.show()

        fig1, axes1 = plt.subplots(1, 1, figsize=(12, 4))
        prikazi_signal(tof_mat, naslov=f'Time-of-Flight\nresolution 1 mm', Fvz=Fvz_tof, enota='distance (mm)', ax=axes1)
        plt.show()

""" Primer izseka 2-4 sekund za vse senzorje 
    a = 2
    b = 4
    fig2, axes2 = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig2.suptitle(f'Vsi senzorji — interval {a}-{b}s', fontsize=15, y=0.98)


    prikazi_signal(gyro_mat_dps, naslov=f'Gyroscope\nresolution 8.75e-3 °/s', startInd=int(a*Fvz_gyro), endInd=int(b*Fvz_gyro), Fvz=Fvz_gyro, enota='rotational speed (dps)', ax=axes2[0])
    prikazi_signal(accel_mat_g, naslov=f'Accelerometer\nresolution 6.125e-5 g-force', startInd=int(a*Fvz_accel), endInd=int(b*Fvz_accel), Fvz=Fvz_accel, enota='acceleration (G-force)', ax=axes2[1])
    prikazi_signal(mag_mat_Gauss, naslov=f'Magnetometer\nresolution 1.5e-3 Gauss', startInd=int(a*Fvz_mag), endInd=int(b*Fvz_mag), Fvz=Fvz_mag, enota='magnetic field (Gauss)', ax=axes2[2])   
    prikazi_signal(tof_mat, naslov=f'Time-of-Flight\nresolution 2 mm', startInd=int(a*Fvz_tof), endInd=int(b*Fvz_tof), Fvz=Fvz_tof, enota='distance (mm)', ax=axes2[3])
   
    axes2[-1].set_xlabel('Čas [s]', fontsize=10)
    fig2.subplots_adjust(hspace=0.6, top=0.90, bottom=0.06)
    #plt.show()

    """