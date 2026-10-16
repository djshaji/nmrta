"""
Hippocampal Episodic Memory & Entorhinal Toroidal Spatial Map.
"""

import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class ToroidalGridCellMap:
    def __init__(self):
        self.scales = [0.2, 0.5, 1.0, 2.5]
        self.orientations = [0.0, np.pi / 3.0, 2.0 * np.pi / 3.0]
        self.wavevectors = []
        for s in self.scales:
            for th in self.orientations:
                u = np.array([np.cos(th), np.sin(th)])
                self.wavevectors.append((s, u))
        assert len(self.wavevectors) == 12

    def encode(self, pos):
        p = np.array(pos, dtype=np.float64)
        features = []
        for s, u in self.wavevectors:
            proj = np.dot(u, p)
            phase = 2.0 * np.pi * proj / s
            features.append(np.cos(phase))
            features.append(np.sin(phase))
        features.append(1.0)
        return np.array(features, dtype=np.float64)

class ContinuousHopfieldMemory:
    def __init__(self, d_key=256, d_val=25, capacity=1000, novelty_threshold=1.5, seed=42):
        self.d_key = d_key
        self.d_val = d_val
        self.capacity = capacity
        self.novelty_threshold = novelty_threshold
        self.beta = 1.0 / np.sqrt(d_key)
        
        self.keys = np.zeros((capacity, d_key), dtype=np.float64)
        self.values = np.zeros((capacity, d_val), dtype=np.float64)
        self.size = 0
        self.write_ptr = 0
        self.total_writes = 0

    def write(self, key, value, kl_surprise):
        if kl_surprise >= self.novelty_threshold:
            self.keys[self.write_ptr] = key
            self.values[self.write_ptr] = value
            self.write_ptr = (self.write_ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            self.total_writes += 1
            return True
        return False

    def retrieve(self, query):
        if self.size == 0:
            return np.zeros(self.d_val)
        active_keys = self.keys[:self.size]
        active_vals = self.values[:self.size]
        scores = self.beta * (active_keys @ query)
        weights = softmax(scores)
        return weights @ active_vals
