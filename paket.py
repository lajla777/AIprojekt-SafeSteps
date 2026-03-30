import numpy as np

class Paket:
    def __init__(self, id: int, ts: float, data:np.ndarray):
        self.id = id
        self.ts = ts
        self.data = np.array(data)

    def read(self):
        return self.data
