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

def signali_skupaj(seznam: list, naslov: str = ''):
    """Nariše vse razpoložljive senzorje v eni figuri."""
    SENSOR_IDS = [0x01, 0x02, 0x03, 0x05]
 
    data = {}
    for sid in SENSOR_IDS:
        try:
            Fvz, matrika = sestavi_podatke(seznam, sid)
            data[sid] = (Fvz, matrika)
        except ValueError:
            pass
 
    if not data:
        print("Ni podatkov za prikaz.")
        return
 
    razpolozljivi = list(data.keys())
    n = len(razpolozljivi)
 
    fig, axes = plt.subplots(n, 1, figsize=(12, 4*n), sharex=False)
    if n == 1:
        axes = [axes]
    fig.suptitle(f'Vsi senzorji — {naslov}', fontsize=13, y=0.98)
 
    for i, sid in enumerate(razpolozljivi):
        Fvz, matrika = data[sid]
        if len(matrika) == 0:
            axes[i].text(0.5, 0.5, 'Ni podatkov',
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(SENSOR_NAMES[sid])
            continue
        prikazi_signal(matrika,
                       Fvz=Fvz,
                       naslov=SENSOR_NAMES[sid],
                       y_label=Y_LABELS.get(sid, 'vrednost'),
                       ax=axes[i])
 
    axes[-1].set_xlabel('Čas [s]', fontsize=10)
    fig.subplots_adjust(hspace=0.6, top=0.93, bottom=0.06)
    plt.show()

if __name__ == "__main__":
    print("VIZUALIZACIJA PODATKOV")
 
    ime_datoteke = input("Ime .bin datoteke: ").strip()
    packets = decode_file(ime_datoteke)
    seznam  = v_pakete(packets)
 
    # Pripravi vse senzorje vnaprej
    data = {}
    for sid in [0x01, 0x02, 0x03, 0x05]:
        try:
            Fvz, matrika = sestavi_podatke(seznam, sid)
            data[sid] = (Fvz, matrika)
            print(f"  {SENSOR_NAMES[sid]:15s}: Fvz={Fvz:.1f} Hz,  vzorcev={len(matrika)}")
        except ValueError as e:
            print(f"  {SENSOR_NAMES.get(sid, hex(sid)):15s}: preskočen — {e}")
 
    MENI = {
        '1': 'Vsi senzorji skupaj',
        '2': 'Gyroscope',
        '3': 'Accelerometer',
        '4': 'Magnetometer',
        '5': 'ToF senzor',
        '0': 'Izhod',
    }
    SID_MAP = {'2': 0x01, '3': 0x02, '4': 0x03, '5': 0x05}
 
    while True:
        print("\nKateri signal želiš prikazati?")
        for k, v in MENI.items():
            print(f"  {k}: {v}")
 
        izbira = input("Izberi možnost: ").strip()
 
        if izbira == '0':
            break
 
        elif izbira == '1':
            signali_skupaj(seznam, naslov=ime_datoteke)
 
        elif izbira in SID_MAP:
            sid = SID_MAP[izbira]
            if sid not in data:
                print(f"Ni podatkov za {SENSOR_NAMES[sid]}.")
                continue
 
            Fvz, matrika = data[sid]
 
            # Celoten signal
            prikazi_signal(matrika,
                           Fvz=Fvz,
                           naslov=SENSOR_NAMES[sid],
                           y_label=Y_LABELS[sid])
 
            # Odsek 0–3 s
            prikazi_signal(matrika,
                           Fvz=Fvz,
                           naslov=f"{SENSOR_NAMES[sid]} — odsek 0–3 s",
                           y_label=Y_LABELS[sid],
                           startInd=0,
                           endInd=int(Fvz * 3))
 
        else:
            print("Neveljavna izbira.")
