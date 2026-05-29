import numpy as np
import torch
import sys
import os

from pathlib import Path

from decoder.decode import decode_file

from tools.visualization import sestavi_podatke, v_pakete
from tools.label_tool import calibrate_mag, orientation, pair_angle_distance

from src.models.tof.model import ObstacleCNN

WINDOW_SEC = 1.0    
STRIDE_SEC = 0.5
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

if len(sys.argv) > 1:
    bin_path = sys.argv[1]
else:
    bin_path = input("Enter .BIN file path to analyse: ").strip()
    if not bin_path:
        print("No file provided. Exiting.")
        sys.exit(1)

if not os.path.isfile(bin_path):
    print(f"File not found: {bin_path}")
    sys.exit(1)

print(f"\nRunning inference on: {bin_path}\n")

LABEL_NAMES = [
    "no_obstacle",
    "obstacle_left",
    "obstacle_right",
    "obstacle_center",
    "obstacle_right_left",
]

model = ObstacleCNN(num_classes=len(LABEL_NAMES))
MODEL_PATH = Path(__file__).resolve().parent / "best_model.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

packets  = decode_file(bin_path)
seznam   = v_pakete(packets)

Fvz,      tof_matrix = sestavi_podatke(seznam, 0x05)
gyro_fvz, gyro  = sestavi_podatke(seznam, 0x01)
_,        accel = sestavi_podatke(seznam, 0x02)
_,        mag = sestavi_podatke(seznam, 0x03)


mag = calibrate_mag(mag)
Q, yaw, pitch, roll = orientation(
    gyro, accel, mag,
    gyro_fvz=gyro_fvz,
    accel_fvz=gyro_fvz,
    mag_fvz=gyro_fvz,
)

yaw = yaw - yaw[0]
yaw = (yaw + 180) % 360 - 180   


pairs = pair_angle_distance(yaw, tof_matrix, gyro_fvz=gyro_fvz, tof_fvz=Fvz)
pairs = np.array(pairs, dtype=np.float32)
pairs = np.nan_to_num(pairs, nan=0.0)   


def normalize(ad: np.ndarray) -> np.ndarray:
    ad        = ad.copy()
    angle_rad = np.deg2rad(ad[:, 0])
    out       = np.zeros((len(ad), 3), dtype=np.float32)
    out[:, 0] = np.sin(angle_rad)
    out[:, 1] = np.cos(angle_rad)
    out[:, 2] = np.clip(ad[:, 1], 0, 4000) / 4000.0
    return out


T = int(WINDOW_SEC * Fvz)
stride = int(STRIDE_SEC * Fvz)

if len(pairs) < T:
    print(f"Not enough data: need {T} samples, got {len(pairs)}.")
    sys.exit(0)

predictions = []

for start in range(0, len(pairs) - T + 1, stride):
    window = normalize(pairs[start : start + T])
    x      = torch.tensor(window, dtype=torch.float32)
    x      = x.unsqueeze(0).to(DEVICE)                 

    with torch.no_grad():
        logits = model(x)
        pred   = torch.argmax(logits, dim=1).item()

    predictions.append(pred)

print("Predicted class indices:", predictions)
print("Predicted labels:       ", [LABEL_NAMES[p] for p in predictions])