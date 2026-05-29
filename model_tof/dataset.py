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

class SweepDataset(Dataset):
    def __init__(self, json_paths, window_sec=2.0, tof_fvz=20, stride_sec=0.5):
        self.T = int(window_sec * tof_fvz)  
        stride = int(stride_sec * tof_fvz)  
        self.samples = []

        for path in json_paths:
            with open(path) as f:
                segments = json.load(f)

            for seg in segments:
                if seg["angle_distance"] is None:
                    continue

                ad = np.array(seg["angle_distance"], dtype=np.float32)
                ad = np.nan_to_num(ad, nan=0.0)
                label = LABEL_IDX[seg["label"]]

                for start in range(0, len(ad) - self.T + 1, stride):
                    window = ad[start:start + self.T].copy()
                    window = self._normalize(window)
                    self.samples.append((window, label))

        print(f"Skupaj oken: {len(self.samples)}")
        self._print_distribution()

    def _normalize(self, ad):
        ad[:, 0] /= 180.0                         
        ad[:, 1] = np.clip(ad[:, 1], 0, 4000) / 4000.0  
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
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)