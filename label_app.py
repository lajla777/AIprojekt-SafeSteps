import os
import sys
import matplotlib.pyplot as plt

from decode import decode_file
from visualisation import sestavi_podatke, v_pakete
from label_tool import LabelTool

LABELS_DIR = "labels"


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
    os.makedirs(LABELS_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(bin_path))[0] 
    filename = f"labels_{stem}.json"
    return os.path.join(LABELS_DIR, filename)


if __name__ == "__main__":
    bin_path = get_bin_path()
    save_path = make_save_path(bin_path)

    print(f"Loading:   {bin_path}")
    print(f"Saving to: {save_path}")

    packets = decode_file(bin_path)
    seznam = v_pakete(packets)

    Fvz, matrix = sestavi_podatke(seznam, 0x05)
    


    tool = LabelTool(matrix, Fvz, save_path)

    plt.show()

    tool.save()