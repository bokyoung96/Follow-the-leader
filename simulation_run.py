"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
import json
import logging
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from abc import ABC, abstractmethod

from simulation_weights import *
from simulation_performance import *


# LOCATE DIRECTORY
def locate_dir(dir_name):
    path = Path(dir_name)
    path.mkdir(parents=True, exist_ok=True)


locate_dir("./LOGGER/")
locate_dir("./DATA_CM_2006/")

# LOCATE TELEGRAM API BOT
with open('simulation_telegram_api.json', 'r') as f:
    info = json.load(f)

TOKEN = info["TOKEN"]
chat_id = info["chat_id"]


class RunAbstr(ABC):
    """
    <DESCRIPTION>
    Abstract method of Run for CM-2006 and FL methods.
    """
    @property
    @abstractmethod
    def shares_out_sample(self):
        """
        <DESCRIPTION>
        Extract the invested shares in out-sample under the condition of same stocks as in-sample.
        """
        pass

    @property
    @abstractmethod
    def in_sample_replica_idx(self):
        """
        <DESCRIPTION>
        Calculate the time-series replicated index value at in-sample.
        """
        pass

    @property
    @abstractmethod
    def out_sample_replica_idx(self):
        """
        <DESCRIPTION>
        Calculate the time-series replicated index value at out-sample.
        """
        pass

    @property
    @abstractmethod
    def replica_idx(self):
        """
        <DESCRIPTION>
        Calculate the time-series replicated index value at whole sample.
        """
        pass

    @abstractmethod
    def run_simulation(self):
        """
        <DESCRIPTION>
        Run simulation, appending performance measures for in-sample and out-sample.
        Main usage of the function is for iterating process in simulation.
        """
        pass

    @abstractmethod
    def run(self):
        """
        <DESCRIPTION>
        Run simulation, appending performance measures for the whole sample.
        Main usage of the function is for single process in simulation.
        """
        pass

    @abstractmethod
    def plots(self):
        """
        <DESCRIPTION>
        Plot original index and replicated index.
        Main usage of the function is for single process after run().
        """
        pass


class RunCM(RunAbstr, Weights):
    def __init__(self,
                 N: int = 100,
                 T: int = 1000,
                 k: int = 5,
                 EV: float = 0.999,
                 min_R2: float = 0.8):
        super().__init__(N, T, k, EV, min_R2)
        """
        <DESCRIPTION>
        Run simulation under CM-2006 method.

        <PARAMETER>
        Same as MethodCM class.

        <CONSTRUCTOR>
        self.opt_weights, self.opt_res: Optimal results from self.optimize() in Weights class.
        self.indiv_inv: T/F determination for whether only 1 stock is invested or not.
        
        <NOTIFICATION>
        Function explanation is written in abstract method above.
        """
        if self.shares_n == 1:
            self.opt_weights = self.optimize()
            self.opt_res = None
            self.indiv_inv = True
        else:
            self.opt_weights, self.opt_res = self.optimize()
            self.indiv_inv = False

    @property
    def shares_out_sample(self) -> np.ndarray:
        return self.out_sample.values[self.get_matched_rows()]

    @property
    def in_sample_replica_idx(self):
        if self.shares_n == 1:
            return self.shares
        else:
            return np.dot(self.opt_weights, self.shares)

    @property
    def out_sample_replica_idx(self):
        if self.shares_n == 1:
            return self.shares_out_sample.flatten()
        else:
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
                    self.shares_n,
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


def run(iters: int = 1000, report_interval: int = 10):
    """
    <DESCRIPTION>
    Run under iteration.
    Logs will be saved with raw_attempts and attempts.
    For each pre-defined intervals, telegram message will be sent to the API bot.
    Data will be saved as pkl file.
    """
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    msres_in = []
    msres_out = []

    attempts = 0
    raw_attempts = 0
    logging.basicConfig(filename='./LOGGER/LOGGER_999.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("RunCM.run_simulation")
    while attempts <= iters:
        run = RunCM()
        if (run.indiv_inv) or (run.opt_res.success):
            msre_in, msre_out = run.run_simulation()
            msres_in.append(msre_in)
            msres_out.append(msre_out)
            logger.info(
                "RAW {} / ATTEMPT {}: SUCCESSS".format(raw_attempts + 1, attempts + 1))
            print("ATTEMPT {} SUCCESS. MOVING ON...".format(attempts + 1))
            raw_attempts += 1
            attempts += 1

            if (attempts + 1) % report_interval == 0:
                message = f"ATTEMPT {attempts + 1} COMPLETED. (0.999)"
                params = {
                    'chat_id': chat_id,
                    'text': message
                }
                requests.get(url, params=params)
                pd.DataFrame(msres_in).to_pickle(
                    "./DATA_CM_2006_999/EV999_MSRES_IN_{}.pkl".format(attempts + 1))
                pd.DataFrame(msres_out).to_pickle(
                    "./DATA_CM_2006_999/EV999_MSRES_OUT_{}.pkl".format(attempts + 1))
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
    pd.DataFrame(msres_in).to_pickle("./EV999_MSRES_IN.pkl")
    pd.DataFrame(msres_out).to_pickle("./EV999_MSRES_OUT.pkl")
