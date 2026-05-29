import json
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_IDX = {
    "no_obstacle":        0,
    "obstacle_left":      1,
    "obstacle_right":     2,
    "obstacle_center":    3,
    "obstacle_right_left":4,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_IDX.items()}

SYMMETRY_MAP = {
    "obstacle_left":       "obstacle_right",
    "obstacle_right":      "obstacle_left",
    "no_obstacle":         "no_obstacle",
    "obstacle_center":     "obstacle_center",
    "obstacle_right_left": "obstacle_right_left",
}


class SweepDataset(Dataset):
    def __init__(
        self,
        json_paths,
        window_sec=1.0,   
        tof_fvz=25,       
        stride_sec=0.25,  
        augment=True,
    ):
        self.T = int(window_sec * tof_fvz)
        stride = int(stride_sec * tof_fvz)
        self.samples = []
        self.augment = augment

        for path in json_paths:
            with open(path) as f:
                segments = json.load(f)

            for seg in segments:
                if not seg.get("angle_distance"):
                    continue

                ad = np.array(seg["angle_distance"], dtype=np.float32)
                ad = np.nan_to_num(ad, nan=0.0)

                label_name = seg["label"]
                if label_name not in LABEL_IDX:
                    continue
                label = LABEL_IDX[label_name]

                for start in range(0, len(ad) - self.T + 1, stride):
                    window = self._normalize(ad[start : start + self.T])
                    self.samples.append((window, label))

                    if self.augment and label_name in SYMMETRY_MAP:
                        aug_label  = LABEL_IDX[SYMMETRY_MAP[label_name]]
                        aug_window = self._mirror(window)
                        self.samples.append((aug_window, aug_label))


                    if self.augment and label_name in ("no_obstacle", "obstacle_center"):
                        for _ in range(2):
                            noise = 0.01 if label_name == "no_obstacle" else 0.02
                            self.samples.append((self._jitter(window, dist_std=noise), label))
                        if label_name == "obstacle_center":
                            self.samples.append((self._mirror(self._jitter(window, dist_std=0.02)), label))

        print(f"Total samples: {len(self.samples)}")


    def _normalize(self, ad):
        ad = ad.copy()
        angle_rad = np.deg2rad(ad[:, 0])
        out = np.zeros((len(ad), 3), dtype=np.float32)
        out[:, 0] = np.sin(angle_rad)   # sin(θ)
        out[:, 1] = np.cos(angle_rad)   # cos(θ)
        out[:, 2] = np.clip(ad[:, 1], 0, 4000) / 4000.0
        return out

    def _jitter(self, window, dist_std=0.02):
        out = window.copy()
        out[:, 2] += np.random.normal(0, dist_std, size=out[:, 2].shape).astype(np.float32)
        out[:, 2]  = np.clip(out[:, 2], 0.0, 1.0)
        delta = np.deg2rad(np.random.uniform(-5, 5))
        sin_d, cos_d = np.sin(delta), np.cos(delta)
        sin_orig = out[:, 0].copy()
        cos_orig = out[:, 1].copy()
        out[:, 0] = sin_orig * cos_d + cos_orig * sin_d   
        out[:, 1] = cos_orig * cos_d - sin_orig * sin_d  
        return out

    def _mirror(self, window):
        mirrored = window.copy()
        mirrored[:, 0] = -mirrored[:, 0]  
        return mirrored

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)