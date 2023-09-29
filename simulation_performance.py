"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
from simulation_weights_CM import *


class Performance:
    def __init__(self, original_idx: np.ndarray, replicated_idx: np.ndarray):
        """
        <DESCRIPTION>
        Performance calculation for simulation methods.

        <PARAMETER>
        original_idx: Original index generated from equal-weight investing.
        replicated_idx: Replicated index generated from CM-2006 or FL method.

        <CONSTRUCTOR>
        original_idx, replicated_idx: Inputs from parameters.
        """
        self.original_idx = original_idx
        self.replicated_idx = replicated_idx

    @property
    def perf_error(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Calculate the difference between original index and replciated index.
        """
        return self.original_idx - self.replicated_idx

    @property
    def perf_mean(self) -> float:
        """
        <DESCRIPTION>
        Calculate mean of the difference.
        """
        return np.mean(self.perf_error)

    @property
    def perf_stdev(self) -> float:
        """
        <DESCRIPTION>
        Calculate standard deviation of the difference.
        """
        return np.std(self.perf_error, ddof=1)

    @property
    def perf_mad(self) -> float:
        """
        <DESCRIPTION>
        Calculate MAD of the difference.
        """
        return np.mean(np.abs(self.perf_error - self.perf_mean))

    @property
    def perf_supmod(self) -> float:
        """
        <DESCRIPTION>
        Calculate SUPMOD of the difference.
        """
        return np.max(np.abs(self.perf_error))

    @property
    def perf_mse(self) -> float:
        """
        <DESCRIPTION>
        Calculate MSE of the difference.
        """
        return np.mean(np.sqrt(np.sum(self.perf_error ** 2)))

    @property
    def perf_corr(self) -> float:
        """
        <DESCRIPTION>
        Calculate correlation of the difference.
        """
        return np.corrcoef(self.original_idx, self.replicated_idx)[0, 1]
