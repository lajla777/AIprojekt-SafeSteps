import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decoder.decode import decode_file
from tools.visualization import sestavi_podatke, v_pakete
from collections import Counter

from tools.label_tool import LabelTool

LABELS_DIR = PROJECT_ROOT / "labels"


def get_bin_path():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter path to .bin file: ").strip()

    path = path.strip('"').strip("'")

    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    return path
def make_save_path(bin_path):
    LABELS_DIR.mkdir(exist_ok=True)

    stem = Path(bin_path).stem
    filename = f"labels_{stem}.json"

    return LABELS_DIR / filename

if __name__ == "__main__":
    bin_path = get_bin_path()
    save_path = make_save_path(bin_path)

    print(f"Loading:   {bin_path}")
    print(f"Saving to: {save_path}")

    packets = decode_file(bin_path)

    if not packets:
        print("Error: No packets could be parsed from the file")
        sys.exit(1)

    try:
        seznam = v_pakete(packets)

        Fvz, tof_matrix = sestavi_podatke(seznam, 0x05)

    except ValueError as e:
        print(f"\nERROR loading sensor data: {e}")

        counts = Counter()

        for p in seznam:
            counts[p.id] += 1

        print("\nPacket IDs found:")
        for pid, cnt in sorted(counts.items()):
            print(f"  ID 0x{pid:02X}: {cnt} packets")

        sys.exit(1)

    tof_duration = len(tof_matrix) / Fvz

    print(
        f"ToF: {len(tof_matrix)} samples @ "
        f"{Fvz:.2f} Hz ({tof_duration:.2f}s)"
    )

    signals = {"ToF (mm)": (tof_matrix, Fvz) }
    imu_signals = {}



    for sid, key, name in [
        (0x01, "gyro", "Gyro (°/s)"),
        (0x02, "accel", "Accel (g)"),
        (0x03, "mag", "Mag (Gauss)"),
    ]:
        try:
            fvz_s, mat = sestavi_podatke(seznam, sid)

            signals[name] = (mat, fvz_s)
            imu_signals[key] = (mat, fvz_s)

            print(
                f"{name}: {len(mat)} samples @ "
                f"{fvz_s:.2f} Hz "
                f"({len(mat) / fvz_s:.2f}s)"
            )

        except ValueError as e:
            print(f"{name}: not available — {e}")

    tool = LabelTool(
        signals,
        save_path,
        imu_signals=imu_signals
    )

    plt.show()

    tool.save()
