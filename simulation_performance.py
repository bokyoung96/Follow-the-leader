"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
from simulation_weights import *


class Performance:
    def __init__(self, original_idx: np.ndarray, replicated_idx: np.ndarray):
        self.original_idx = original_idx
        self.replicated_idx = replicated_idx

    @property
    def perf_error(self) -> np.ndarray:
        return self.original_idx - self.replicated_idx

    @property
    def perf_mean(self) -> float:
        return np.mean(self.perf_error)

    @property
    def perf_stdev(self) -> float:
        return np.std(self.perf_error, ddof=1)

    @property
    def perf_mad(self) -> float:
        return np.mean(np.abs(self.perf_error - self.perf_mean))

    @property
    def perf_supmod(self) -> float:
        return np.max(np.abs(self.perf_error))

    @property
    def perf_mse(self) -> float:
        return np.mean(np.sqrt(np.sum(self.perf_error ** 2)))

    @property
    def perf_corr(self) -> float:
        return np.corrcoef(self.original_idx, self.replicated_idx)[0, 1]
