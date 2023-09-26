import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from simulation_weights import *
from simulation_performance import *


class RunAbstr(ABC):
    @property
    @abstractmethod
    def shares_out_sample(self):
        pass

    @property
    @abstractmethod
    def in_sample_replica_idx(self):
        pass

    @property
    @abstractmethod
    def out_sample_replica_idx(self):
        pass

    @property
    @abstractmethod
    def replica_idx(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def plots(self):
        pass


class RunCM(RunAbstr, Weights):
    def __init__(self,
                 N: int = 100,
                 T: int = 1000,
                 k: int = 5,
                 EV: float = 0.999,
                 min_R2: float = 0.8):
        super().__init__(N, T, k, EV, min_R2)
        self.opt_weights, self.opt_res = self.optimize()

    @property
    def shares_out_sample(self) -> np.ndarray:
        return self.out_sample.values[self.get_matched_rows()]

    @property
    def in_sample_replica_idx(self):
        return np.dot(self.opt_weights, self.shares)

    @property
    def out_sample_replica_idx(self):
        return np.dot(self.opt_weights, self.shares_out_sample)

    @property
    def replica_idx(self):
        return np.hstack((self.in_sample_replica_idx, self.out_sample_replica_idx))

    def run(self):
        perf = Performance(np.array(self.idx), self.replica_idx)
        msres = [len(self.F_pca),
                 len(self.shares),
                 perf.perf_mean,
                 perf.perf_stdev,
                 perf.perf_mad,
                 perf.perf_supmod,
                 perf.perf_mse]
        res = pd.DataFrame(msres, columns=['PERF_MSRE'], index=['Nfactors',
                                                                'Nleaders',
                                                                'MEAN',
                                                                'STDEV',
                                                                'MAD',
                                                                'SUPMOD',
                                                                'MSE'])
        return res

    def plots(self):
        idx = self.idx
        idx_method = self.replica_idx.flatten()

        plt.figure(figsize=(15, 5))
        plt.plot(idx, label='Original index')
        plt.plot(idx_method, label='Replicated index')
        plt.axvline(x=500, color='red', linestyle='--',
                    label='In & Out sample division')

        plt.title(
            'Original index versus Replicated index in optimial weights under PCA EV cutoff {}'.format(self.EV))
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc='best')

        plt.show()


if __name__ == "__main__":
    run = RunCM()
    res = run.run()
    run.plots()
    print(res)
