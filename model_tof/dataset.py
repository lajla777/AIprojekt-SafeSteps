import json
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_IDX = {
    "no_obstacle": 0,
    "obstacle_left": 1,
    "obstacle_right": 2,
    "obstacle_center": 3,
    "obstacle_right_left": 4,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_IDX.items()}

SYMMETRY_MAP = {
    "obstacle_left": "obstacle_right",
    "obstacle_right": "obstacle_left",
    "obstacle_center": "obstacle_center",
    "no_obstacle": "no_obstacle",
    "obstacle_right_left": "obstacle_right_left",  
}

class SweepDataset(Dataset):
    def __init__(self, json_paths, window_sec=2.0, tof_fvz=20, stride_sec=0.5, augment=True):
        self.T = int(window_sec * tof_fvz)
        stride = int(stride_sec * tof_fvz)
        self.samples = []
        self.augment = augment

        for path in json_paths:
            with open(path) as f:
                segments = json.load(f)

            for seg in segments:
                if seg.get("angle_distance") is None:
                    continue

                ad = np.array(seg["angle_distance"], dtype=np.float32)
                ad = np.nan_to_num(ad, nan=0.0)

                label_name = seg["label"]
                label = LABEL_IDX[label_name]

                for start in range(0, len(ad) - self.T + 1, stride):
                    window = ad[start:start + self.T].copy()
                    window = self._normalize(window)
                    self.samples.append((window, label))

                    if self.augment and label_name in SYMMETRY_MAP:
                        aug_label_name = SYMMETRY_MAP[label_name]
                        aug_label = LABEL_IDX[aug_label_name]

                        aug_window = self._mirror(window.copy())

                        self.samples.append((aug_window, aug_label))

        print(f"Skupaj oken: {len(self.samples)}")
        self._print_distribution()

    def _normalize(self, ad):
        ad[:, 0] = ad[:, 0] / 180.0

        ad[:, 1] = np.clip(ad[:, 1], 0, 4000) / 4000.0
        return ad

    def _mirror(self, ad):
        ad = ad.copy()
        ad[:, 0] = -ad[:, 0]
        return ad

    def _print_distribution(self):
        from collections import Counter
        counts = Counter(y for _, y in self.samples)
        idx_to_label = {v: k for k, v in LABEL_IDX.items()}

        print("Porazdelitev razredov:")
        for idx, count in sorted(counts.items()):
            print(f"  {idx_to_label[idx]}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)