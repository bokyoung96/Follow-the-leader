"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
import json
import pickle
import logging
import requests
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from simulation_weights import *
from simulation_performance import *


with open('simulation_telegram_api.json', 'r') as f:
    info = json.load(f)

TOKEN = info["TOKEN"]
chat_id = info["chat_id"]


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
    def run_simulation(self):
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
                 EV: float = 0.99,
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

    def run_simulation(self):
        perf_in = Performance(np.array(self.idx_in_sample),
                              self.in_sample_replica_idx)
        perf_out = Performance(np.array(self.idx_out_sample),
                               self.out_sample_replica_idx)

        msres_in = [len(self.F_pca),
                    len(self.shares),
                    perf_in.perf_mean,
                    perf_in.perf_stdev,
                    perf_in.perf_mad,
                    perf_in.perf_supmod,
                    perf_in.perf_mse]
        msres_out = [perf_out.perf_mean,
                     perf_out.perf_stdev,
                     perf_out.perf_mad,
                     perf_out.perf_supmod,
                     perf_out.perf_mse,
                     perf_out.perf_corr]
        return msres_in, msres_out

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


def run(iters: int = 6, report_interval: int = 2):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    msres_in = []
    msres_out = []

    attempts = 0
    raw_attempts = 0
    logging.basicConfig(filename='LOGGER.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("RunCM.run_simulation")
    while attempts <= iters:
        run = RunCM()
        if run.opt_res.success:
            msre_in, msre_out = run.run_simulation()
            msres_in.append(msre_in)
            msres_out.append(msre_out)
            logger.info(
                "RAW {} / ATTEMPT {}: SUCCESSS".format(raw_attempts + 1, attempts + 1))
            raw_attempts += 1
            attempts += 1

            if (attempts + 1) % report_interval == 0:
                message = f"ATTEMPT {attempts + 1} COMPLETED."
                params = {
                    'chat_id': chat_id,
                    'text': message
                }
                requests.get(url, params=params)
        else:
            logger.warning(
                "RAW {} / ATTEMPT {}: FAILED".format(raw_attempts + 1, attempts + 1))
            raw_attempts += 1
            continue

    with open('LOGGER.log', 'r') as f_log:
        log_contents = f_log.read()

    message = "LOGS: \n\n\n" + log_contents
    params = {
        'chat_id': chat_id,
        'text': message
    }
    requests.get(url, params=params)

    print("***** FINISHED: CHECK TELEGRAM BOT *****")
    return msres_in, msres_out


if __name__ == "__main__":
    msres_in, msres_out = run()
    # pd.DataFrame(msres_in).to_pickle("./EV99_MSRES_IN.pkl")
    # pd.DataFrame(msres_out).to_pickle("./EV99_MSRES_OUT.pkl")
