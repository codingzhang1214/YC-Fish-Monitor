import numpy as np


class FeatureBank:
    def __init__(self, dim: int, capacity: int = 200, momentum: float = 0.9):
        self.dim = dim
        self.capacity = max(1, int(capacity))
        self.momentum = float(np.clip(momentum, 0.0, 1.0))
        self.queue = []
        self.avg = None

    def add(self, feat: np.ndarray):
        if feat is None or feat.size == 0:
            return
        f = feat.astype(np.float32)
        f = f / (np.linalg.norm(f) + 1e-12)
        self.queue.append(f)
        if len(self.queue) > self.capacity:
            self.queue.pop(0)
        if self.avg is None:
            self.avg = f.copy()
        else:
            self.avg = self.momentum * self.avg + (1 - self.momentum) * f
            self.avg = self.avg / (np.linalg.norm(self.avg) + 1e-12)

    def representation(self, k_last: int = 5) -> np.ndarray:
        if self.queue:
            k = min(k_last, len(self.queue))
            rep = np.mean(self.queue[-k:], axis=0)
            rep = rep / (np.linalg.norm(rep) + 1e-12)
            return rep
        return self.avg if self.avg is not None else np.zeros(self.dim, dtype=np.float32)
