"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import datetime
import itertools
import numpy as np
import pandas as pd

from empirical_func import *
from empirical_loader import *
from time_spent_decorator import time_spent_decorator


class PerformanceMeasure:
    def __init__(self, replicated_idx: np.ndarray, original_idx: np.ndarray):
        """
        <DESCRIPTION>
        Performance calculation for empirical methods.

        <PARAMETER>
        original_idx: Original index.
        replicated_idx: Replicated index generated from CM-2006 or FL method.

        <CONSTRUCTOR>
        original_idx, replicated_idx: Inputs from parameters.
        """
        self.replicated_idx = replicated_idx
        self.original_idx = original_idx

    @property
    def perf_error(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Calculate the difference between replicated index and original index.
        """
        return self.replicated_idx - self.original_idx

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


class Performance(DataLoader, Func):
    stopper = None

    def __init__(self,
                 freq_1: int = 125,
                 freq_2: int = 5,
                 date_dir: datetime = datetime.date.today(),
                 method_type: str = 'FL',
                 p_val: float = 0.1,
                 EV: float = 0.99,
                 min_R2: float = 0.8,
                 init_price: int = 100,
                 mkt: str = 'KOSPI200',
                 date: str = 'Y5',
                 idx_weight: str = 'EQ'):
        """
        <DESCRIPTION>
        Performance calculation for empirical methods.

        <PARAMETER>
        Same as MethodRunnerCM, MethodRunnerFL.
        date_dir: Date string for loading directory files.
        method_type: Type of method to use.

        <CONSTRUCTOR>
        stopper: Stop redundant repeats to call original index at iteration.
        """
        super().__init__(mkt, date)
        self.replicated_idx = None
        self.original_idx = None
        self.shares_count = None
        self.shares_count_matched = None
        self.F_nums = None
        self.backtest_period = None

        if Performance.stopper is None:
            Performance.stopper = self.as_empirical(
                idx_weight=idx_weight)[0].pct_change()
        else:
            pass

        self.freq_1 = freq_1
        self.freq_2 = freq_2
        self.date_dir = date_dir
        self.method_type = method_type
        self.p_val = p_val
        self.EV = EV
        self.min_R2 = min_R2
        self.init_price = init_price
        self.mkt = mkt
        self.date = date

        if method_type == 'FL':
            self.dir_global = "RUNNER_FL_{}_{}_{}_p_val_{}".format(self.freq_1,
                                                                   self.freq_2,
                                                                   self.date_dir,
                                                                   self.p_val)
        elif method_type == 'CM':
            self.dir_global = "RUNNER_CM_{}_{}_{}_ev_{}".format(self.freq_1,
                                                                self.freq_2,
                                                                self.date_dir,
                                                                self.EV)

        self.perf_load
        self.perf_msre = PerformanceMeasure(
            self.replicated_idx, self.original_idx)

    @property
    def perf_load(self):
        """
        <DESCRIPTION>
        Load and preprocess datas to be used in performance calculation.
        """
        self.replicated_idx = pd.read_pickle("./{}/replica_{}.pkl".format(self.dir_global,
                                                                          self.date))
        self.original_idx = Performance.stopper
        self.original_idx = self.original_idx[self.freq_1 -
                                              1:self.freq_1 + len(self.replicated_idx)-1]

        self.replicated_idx = self.func_plot_init_price(self.replicated_idx,
                                                        self.init_price).cumsum()
        self.original_idx = self.func_plot_init_price(self.original_idx,
                                                      self.init_price).cumsum()

        self.shares_count = pd.read_pickle("./{}/shares_count_{}.pkl".format(self.dir_global,
                                                                             self.date))
        self.shares_count_matched = pd.DataFrame(np.concatenate(
            [item for item in self.shares_count.values for _ in range(self.freq_2)]))
        self.shares_count_matched = self.func_plot_init_price(
            self.shares_count_matched, np.nan)

        self.F_nums = pd.read_pickle("./{}/F_nums_count_{}.pkl".format(self.dir_global,
                                                                       self.date))
        self.backtest_period = self.original_idx.index

        self.replicated_idx = self.replicated_idx.values.flatten()
        self.original_idx = self.original_idx.values
        self.shares_count_matched = self.shares_count_matched.values
        print("DATA LOAD & PREPROCESS DONE. MOVING ON...")

    def perf_run(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Calculate performance measures.
        """
        msres = np.array([self.F_nums.values.mean(),
                          self.shares_count.values.mean(),
                          self.perf_msre.perf_mean,
                          self.perf_msre.perf_stdev,
                          self.perf_msre.perf_mad,
                          self.perf_msre.perf_supmod,
                          self.perf_msre.perf_mse,
                          self.perf_msre.perf_corr])
        msres = np.round(msres, decimals=4)

        res = pd.DataFrame(msres, columns=['{}'.format(self.freq_2)], index=['Nfactors',
                                                                             'Nleaders',
                                                                             'MEAN',
                                                                             'STDEV',
                                                                             'MAD',
                                                                             'SUPMOD',
                                                                             'MSE',
                                                                             'CORR'])
        return res


def perf_each(method_type: str = 'FL',
              days: int = 2) -> dict:
    """
    <DESCRIPTION>
    Iterate performance calculation for each methods.
    """
    res_dict = {}

    freq_1s = [125, 250, 375]
    freq_2s = [5, 10, 15, 20]
    for val_1, val_2 in itertools.product(freq_1s, freq_2s):
        freq_1 = val_1
        freq_2 = val_2
        date_dir = datetime.date.today() - datetime.timedelta(days=days)
        perf_fl = Performance(freq_1=freq_1,
                              freq_2=freq_2,
                              date_dir=date_dir,
                              method_type=method_type)
        df = perf_fl.perf_run()

        if freq_1 not in res_dict:
            res_dict[freq_1] = []
        res_dict[freq_1].append(df)

    for freq, dfs in res_dict.items():
        df_concat = pd.concat(dfs, axis=1)
        res_dict[freq] = df_concat
        res_dict[freq].index.name = f"IN-SAMPLE {freq}"
        res_dict[freq].columns.name = "OUT-SAMPLE"
        res_dict[freq].columns = pd.MultiIndex.from_product(
            [[method_type], res_dict[freq].columns])
    return res_dict


@time_spent_decorator
def perf_concat():
    """
    <DESCRIPTION>
    Concat results from perf_each.

    <NOTIFICATION>
    Call results by res_merged[freq_1].
    """
    res_merged = {}

    res_fl = perf_each(method_type='FL', days=2)
    res_cm = perf_each(method_type='CM', days=1)
    for key in res_fl.keys():
        res_merged[key] = pd.concat([res_fl[key], res_cm[key]], axis=1)
    return res_merged


if __name__ == "__main__":
    res = perf_concat()
