import numpy as np
import matplotlib.pyplot as plt
from logger import read_file

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
    0x05: "ToF",
}
Y_LABELS = {
    0x01: "Rotational Speed (°/s)",
    0x02: "G-force (g)",
    0x03: "Magnetic Field (Gauss)",
    0x05: "Distance (mm)",
}

def sestavi_podatke(packets, id):
    sensor_packets = [p for p in packets if p.id == id]
    if len(sensor_packets) < 1:
        raise ValueError(f"Not enaugh packets ID {hex(id)}")
    differences = []
    for i in range(len(sensor_packets) - 1):
        diff = sensor_packets[i + 1].ts - sensor_packets[i].ts
        differences.append(diff)
    Tpacket = np.mean(differences)
    Nvz = np.mean([p.data.shape[0] for p in sensor_packets])
    Fvz = Nvz / (Tpacket / 1000)
    resolution = RESOLUTION.get(id, 1)
    matrix = np.vstack([p.data for p in sensor_packets]) * resolution
    return Fvz, matrix

def prikazi_signal(signal, Fvz=1.0, naslov=None, y_label="Vrednost",startInd=None, endInd=None, ax=None, standalone=True):
    part = signal[startInd:endInd]
    start = startInd if startInd is not None else 0
    t = (np.arange(len(part)) + start) / Fvz
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 4))
    if signal.shape[1] == 1:  # ToF - en stolpec
        ax.plot(t, part[:, 0], label='Distance', color='purple', linewidth=0.8)
    else:  # X, Y, Z
        ax.plot(t, part[:, 0], label='X', linewidth=0.8)
        ax.plot(t, part[:, 1], label='Y', linewidth=0.8)
        ax.plot(t, part[:, 2], label='Z', linewidth=0.8)
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    if naslov:
        ax.set_title(naslov)
    ax.grid(True, alpha=0.3)
    if standalone:
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    packets = read_file("stream_1776190334.bin")

    sensor_ids = [0x01, 0x02, 0x03]
    data = {}
    for sid in sensor_ids:
        try:
            Fvz, matrix = sestavi_podatke(packets, sid)
            data[sid] = (Fvz, matrix)
            print(f"{SENSOR_NAMES[sid]}: Fvz = {Fvz:.2f} Hz, vzorcev = {len(matrix)}")
        except ValueError as e:
            print(f"{SENSOR_NAMES[sid]}: preskočen — {e}")
    try:
        Fvz_tof, matrix_tof = sestavi_podatke(packets, 0x05)
        data[0x05] = (Fvz_tof, matrix_tof)
        print(f"{SENSOR_NAMES[0x05]}: Fvz = {Fvz_tof:.2f} Hz, vzorcev = {len(matrix_tof)}")
        has_tof = True
    except ValueError as e:
        print(f"ToF: {e}")
        has_tof = False

    all_ids = sensor_ids + ([0x05] if has_tof else [])
    n = len(all_ids)

    fig1, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    for i, sid in enumerate(all_ids):
        Fvz, matrix = data[sid]
        prikazi_signal(matrix, Fvz=Fvz, naslov=f"{SENSOR_NAMES[sid]}  -  Fvz = {Fvz:.2f} Hz",
            y_label=Y_LABELS[sid], ax=axes[i], standalone=False)
    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(n, 1, figsize=(14, 4 * n))
    for i, sid in enumerate(all_ids):
        Fvz, matrix = data[sid]
        start_idx = int(7.0 * Fvz)
        end_idx = min(int(11.0 * Fvz), len(matrix))
        prikazi_signal(matrix, Fvz=Fvz, naslov=f"{SENSOR_NAMES[sid]} - odsek",
            y_label=Y_LABELS[sid], startInd=start_idx, endInd=end_idx,
            ax=axes2[i], standalone=False)
    plt.tight_layout()
    plt.show()