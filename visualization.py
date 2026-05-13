import numpy as np
from dataclasses import dataclass
from decode import decode_file
import matplotlib.pyplot as plt

RESOLUTION = {
    0x01: 8.75e-3,
    0x02: 6.1035e-5,
    0x03: 1.5e-3,
    0x05: 1
}

SENSOR_NAMES = {
    0x01: "Gyroscope",
    0x02: "Accelerometer",
    0x03: "Magnetometer",
    0x05: "ToF sensor",
}

Y_LABELS = {
    0x01: "Rotational Speed (°/s)",
    0x02: "G-force (g)",
    0x03: "Magnetic Field (Gauss)",
    0x05: "Distance (mm)",
}

@dataclass
class Paket:
    id:   int
    ts:   float
    data: np.ndarray

def v_pakete(packets: list) -> list:
    """Pretvori seznam paketov (iz decode_file) v seznam Paket objektov."""
    seznam = []
    for pkt in packets:
        ts = pkt['timestamp_ms']
        for chunk_id, chunk in pkt['chunks'].items():
            samples = chunk['samples']
            if not samples:
                continue
            arr = np.array(samples)
            seznam.append(Paket(id=chunk_id, ts=ts, data=arr))
    return seznam


def sestavi_podatke(seznam: list, sensor_id: int):
    sensor_paketi = [p for p in seznam if p.id == sensor_id]
    if len(sensor_paketi) < 2:
        raise ValueError(f"Premalo paketov za ID {hex(sensor_id)}")

    differences = []
    for i in range(len(sensor_paketi) - 1):
        diff = sensor_paketi[i + 1].ts - sensor_paketi[i].ts
        differences.append(diff)
    Tpacket = np.mean(differences) / 1000.0    # ms → s
    Nvz = np.mean([p.data.shape[0] for p in sensor_paketi])
    Fvz = Nvz / Tpacket

    resolution = RESOLUTION.get(sensor_id, 1)
    matrika = np.vstack([p.data for p in sensor_paketi]) * resolution

    # ToF: neveljavne vrednosti → NaN
    if sensor_id == 0x05:
        matrika = matrika.astype(np.float32)
        matrika[matrika == 0xFFFF * resolution] = np.nan

    return Fvz, matrika


def prikazi_signal(signal: np.ndarray,
                   Fvz: float = 1.0,
                   naslov: str = None,
                   y_label: str = 'Vrednost',
                   startInd: int = None,
                   endInd: int = None,
                   ax=None):
    
    odsek = signal[startInd:endInd]
    start = startInd if startInd is not None else 0
    t = (np.arange(len(odsek)) + start) / Fvz
 
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 4))
 
    K = odsek.shape[1] if odsek.ndim > 1 else 1
    koordinate = ['X', 'Y', 'Z'] if K == 3 else (['Distance'] if K == 1 else [str(i) for i in range(K)])
    BARVE = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'purple']

    if odsek.ndim == 1 or K == 1:
        y = odsek[:, 0] if odsek.ndim > 1 else odsek
        ax.plot(t, y, label=koordinate[0], color=BARVE[4], linewidth=0.8)
    else:
        for k in range(K):
            ax.plot(t, odsek[:, k], label=koordinate[k],
                    color=BARVE[k % len(BARVE)], linewidth=0.8)

    title = naslov if naslov else ''
    if Fvz is not None and Fvz > 0:
        title += f', sampling rate: {Fvz:.1f} Hz'
    if title:
        ax.set_title(title, fontsize=10)

    ax.set_ylabel(y_label, fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.4)

    if standalone:
        ax.set_xlabel('Čas [s]', fontsize=10)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
 
    # ── Nastavi vhod ──────────────────────────
    # Za eno datoteko:
    files = ['test.bin']
 
    # Za več datotek (odkomentiraj):
    # files = ['Zajemanje podatkov/Oseba na prostem/LOG%03i.BIN' % i for i in range(1, 25)]
 
    SENSOR_IDS = [0x01, 0x02, 0x03, 0x05]
 
    for file in files:
        print(f"\n{'='*50}")
        print(f"Datoteka: {file}")
 
        packets = decode_file(file)
        seznam  = v_pakete(packets)
 
        # Sestavi podatke za vse senzorje
        data = {}
        for sid in SENSOR_IDS:
            try:
                Fvz, matrika = sestavi_podatke(seznam, sid)
                data[sid] = (Fvz, matrika)
                print(f"  {SENSOR_NAMES[sid]:15s}: Fvz={Fvz:.1f} Hz,  vzorcev={len(matrika)}")
            except ValueError as e:
                print(f"  {SENSOR_NAMES.get(sid, hex(sid)):15s}: preskočen — {e}")
 
        if not data:
            print("  Ni podatkov za prikaz.")
            continue
 
        razpolozljivi = list(data.keys())
        n = len(razpolozljivi)
 
        # ── Graf 1: celoten signal ─────────────
        fig1, axes1 = plt.subplots(n, 1, figsize=(14, 4*n), sharex=True)
        if n == 1:
            axes1 = [axes1]
        fig1.suptitle(f'Vsi senzorji — {file}', fontsize=13, y=0.98)
 
        for i, sid in enumerate(razpolozljivi):
            Fvz, matrika = data[sid]
            prikazi_signal(matrika,
                           Fvz=Fvz,
                           naslov=SENSOR_NAMES[sid],
                           y_label=Y_LABELS.get(sid, 'vrednost'),
                           ax=axes1[i])
 
        axes1[-1].set_xlabel('Čas [s]', fontsize=10)
        fig1.subplots_adjust(hspace=0.5, top=0.93, bottom=0.06)
        plt.show()
 
        # ── Graf 2: odsek (npr. 2–4 s) ────────
        a, b = 2.0, 4.0
        fig2, axes2 = plt.subplots(n, 1, figsize=(14, 4*n), sharex=True)
        if n == 1:
            axes2 = [axes2]
        fig2.suptitle(f'Odsek {a}–{b} s — {file}', fontsize=13, y=0.98)
 
        for i, sid in enumerate(razpolozljivi):
            Fvz, matrika = data[sid]
            start_idx = int(a * Fvz)
            end_idx   = min(int(b * Fvz), len(matrika))
            prikazi_signal(matrika,
                           Fvz=Fvz,
                           naslov=SENSOR_NAMES[sid],
                           y_label=Y_LABELS.get(sid, 'vrednost'),
                           startInd=start_idx,
                           endInd=end_idx,
                           ax=axes2[i])
 
        axes2[-1].set_xlabel('Čas [s]', fontsize=10)
        fig2.subplots_adjust(hspace=0.5, top=0.93, bottom=0.06)
        plt.show()
 