"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.preprocessing import MinMaxScaler
from numpy.polynomial.polynomial import Polynomial

from empirical_func import *
from empirical_loader import *
from empirical_plot_params import *
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

    @staticmethod
    def perf_mean_rolling(replicated_idx: np.ndarray, original_idx: np.ndarray) -> float:
        """
        <DESCRIPTION>
        Calculate rolling mean of the difference.
        """
        perf_error = replicated_idx - original_idx
        return np.mean(perf_error)

    @property
    def perf_stdev(self) -> float:
        """
        <DESCRIPTION>
        Calculate standard deviation of the difference.
        """
        return np.std(self.perf_error, ddof=1)

    @staticmethod
    def perf_stdev_rolling(replicated_idx: np.ndarray, original_idx: np.ndarray) -> float:
        """
        <DESCRIPTION>
        Calculate rolling stdev of the difference.
        """
        perf_error = replicated_idx - original_idx
        return np.std(perf_error, ddof=1)

    @property
    def perf_mad(self) -> float:
        """
        <DESCRIPTION>
        Calculate MAD of the difference.
        """
        return np.mean(np.abs(self.perf_error))

    @staticmethod
    def perf_mad_rolling(replicated_idx: np.ndarray, original_idx: np.ndarray) -> float:
        """
        <DESCRIPTION>
        Calculate rolling MAD of the difference.
        """
        perf_error = replicated_idx - original_idx
        return np.mean(np.abs(perf_error))

    @property
    def perf_supmod(self) -> float:
        """
        <DESCRIPTION>
        Calculate SUPMOD of the difference.
        """
        return np.max(np.abs(self.perf_error))

    @staticmethod
    def perf_supmod_rolling(replicated_idx: np.ndarray, original_idx: np.ndarray) -> float:
        """
        <DESCRIPTION>
        Calculate rolling SUPMOD of the difference.
        """
        perf_error = replicated_idx - original_idx
        return np.max(np.abs(perf_error))

    @property
    def perf_mse(self) -> float:
        """
        <DESCRIPTION>
        Calculate MSE of the difference.
        """
        return np.sqrt(np.sum(self.perf_error ** 2)) / self.perf_error.shape[0]

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
                 method_type: str = 'FL',
                 p_val: float = 0.1,
                 EV: float = 0.9,
                 min_R2: float = 0.8,
                 num: int = 100,
                 init_price: int = 100,
                 mkt: str = 'KOSPI200',
                 date: str = 'Y15',
                 idx_weight: str = 'EQ',
                 start_date: str = '2011-01-01'):
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
        self.weights_count = None
        self.F_nums = None
        self.backtest_period = None

        if Performance.stopper is None:
            Performance.stopper = self.as_empirical(
                idx_weight=idx_weight)[0].pct_change()
        else:
            pass

        self.freq_1 = freq_1
        self.freq_2 = freq_2
        self.method_type = method_type
        self.p_val = p_val
        self.EV = EV
        self.min_R2 = min_R2
        self.num = num
        self.init_price = init_price
        self.mkt = mkt
        self.date = date
        self.start_date = start_date

        if method_type == 'FL':
            self.dir_global = "RUNNER_FL_{}_{}_p_val_{}".format(self.freq_1,
                                                                self.freq_2,
                                                                self.p_val)
            self.dir_main = "RUNNER_FL_{}_p_val_{}".format(self.date,
                                                           self.p_val)
            if self.mkt == 'KOSDAQ150':
                self.dir_main = self.dir_main + "_KOSDAQ150"
        elif method_type == 'CM':
            self.dir_global = "RUNNER_CM_{}_{}_ev_{}".format(self.freq_1,
                                                             self.freq_2,
                                                             self.EV)
            self.dir_main = "RUNNER_CM_{}_ev_{}".format(self.date,
                                                        self.EV)
        elif method_type == 'NV':
            self.dir_global = "RUNNER_NV_{}_{}_num_{}".format(self.freq_1,
                                                              self.freq_2,
                                                              self.num)
            self.dir_main = "RUNNER_NV_{}_num_{}".format(self.date,
                                                         self.num)
        elif method_type == 'FL_REAL':
            self.dir_global = "RUNNER_FL_{}_{}_p_val_{}".format(self.freq_1,
                                                                self.freq_2,
                                                                self.p_val)
            self.dir_main = "RUNNER_FL_{}_p_val_{}_REAL".format(self.date,
                                                                self.p_val)

        self.perf_load
        self.perf_msre = PerformanceMeasure(
            self.replicated_idx, self.original_idx)

    @property
    def perf_load(self):
        """
        <DESCRIPTION>
        Load and preprocess datas to be used in performance calculation.
        """
        self.replicated_idx = pd.read_pickle("./{}/{}/replica_{}.pkl".format(self.dir_main,
                                                                             self.dir_global,
                                                                             self.date))
        self.original_idx = Performance.stopper
        self.original_idx = self.original_idx[self.start_date:]
        self.original_idx = self.original_idx[:len(self.replicated_idx)]

        self.replicated_idx = (1 + self.func_plot_init_price(self.replicated_idx,
                                                             self.init_price)).cumprod() - 1
        self.original_idx = (1 + self.func_plot_init_price(self.original_idx,
                                                           self.init_price)).cumprod() - 1

        self.shares_count = pd.read_pickle("./{}/{}/shares_count_{}.pkl".format(self.dir_main,
                                                                                self.dir_global,
                                                                                self.date))
        self.shares_count_matched = pd.DataFrame(np.concatenate(
            [item for item in self.shares_count.values for _ in range(self.freq_2)]))
        self.shares_count_matched = self.func_plot_init_price(
            self.shares_count_matched, np.nan)

        self.weights_count = pd.read_pickle("./{}/{}/weights_count_{}.pkl".format(self.dir_main,
                                                                                  self.dir_global,
                                                                                  self.date))
        self.weights_save = pd.read_pickle("./{}/{}/weights_save_{}.pkl".format(self.dir_main,
                                                                                self.dir_global,
                                                                                self.date))

        # self.mktcap = pd.read_pickle(
        #     "./{}_TRANSACTION_COST/{}_MKTCAP_{}.pkl".format(self.mkt,
        #                                                     self.mkt,
        #                                                     self.date))

        if self.method_type == 'NV':
            self.F_nums = pd.DataFrame([0] * len(self.shares_count))
        else:
            self.F_nums = pd.read_pickle("./{}/{}/F_nums_count_{}.pkl".format(self.dir_main,
                                                                              self.dir_global,
                                                                              self.date))

        self.trans_cost_raw = pd.read_pickle(
            f'./{self.mkt}_TRANSACTION_COST/{self.mkt}_TRANSACTION_COST_{self.date}.pkl')
        self.trans_cost_raw = self.trans_cost_raw.loc[self.start_date:'2022-12-31']
        self.trans_cost_raw = self.trans_cost_raw[self.weights_save.columns]

        self.backtest_period = self.original_idx.index.tolist()
        self.backtest_period[0] = self.original_idx.index[1] - \
            pd.DateOffset(days=1)

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

    def perf_rolling(self) -> pd.DataFrame:
        """
        <DESCIRPTION>
        Caculate performance measures for each window.
        """
        replicated_idx = self.replicated_idx[1:]
        original_idx = self.original_idx[1:]

        mad_rolling = np.array([self.perf_msre.perf_mad_rolling(replicated_idx[i:i+self.freq_2],
                               original_idx[i:i+self.freq_2])for i in range(0, len(replicated_idx), self.freq_2)])
        mad_rolling = pd.DataFrame(np.round(mad_rolling, decimals=4))

        res = pd.concat([self.F_nums, self.shares_count, mad_rolling], axis=1)
        res.columns = ['Factors', 'Leaders', 'MAD']
        res.columns.name = f'{self.freq_1} / {self.freq_2}'

        if self.method_type == 'FL' or self.method_type == 'NV' or self.method_type == "FL_REAL":
            res.columns = pd.MultiIndex.from_product([[self.method_type],
                                                      res.columns])
        elif self.method_type == 'CM':
            res.columns = pd.MultiIndex.from_product([[f'{self.method_type}_{self.EV}'],
                                                      res.columns])
        res.index.name = 'Window'
        return res

    def perf_rolling_values(self) -> pd.DataFrame:
        """
        <DESCIRPTION>
        Caculate performance measures rolled for each window.
        """
        replicated_idx = self.replicated_idx[1:]
        original_idx = self.original_idx[1:]

        def rolling_values(replicated_idx, original_idx, freq_2, perf_func):
            return pd.DataFrame([perf_func(replicated_idx[i:i+freq_2],
                                           original_idx[i:i+freq_2])
                                 for i in range(0, len(replicated_idx), freq_2)])
        mean_rolling = rolling_values(
            replicated_idx, original_idx, self.freq_2, self.perf_msre.perf_mean_rolling)
        stdev_rolling = rolling_values(
            replicated_idx, original_idx, self.freq_2, self.perf_msre.perf_stdev_rolling)
        mad_rolling = rolling_values(
            replicated_idx, original_idx, self.freq_2, self.perf_msre.perf_mad_rolling)
        supmod_rolling = rolling_values(
            replicated_idx, original_idx, self.freq_2, self.perf_msre.perf_supmod_rolling)

        res = pd.concat([self.F_nums, self.shares_count,
                         mean_rolling, stdev_rolling,
                         mad_rolling, supmod_rolling], axis=1)
        res.columns = ['Factors', 'Leaders',
                       'MEAN', 'STDEV',
                       'MAD', 'SUPMOD']
        res.columns.name = f'{self.freq_1} / {self.freq_2}'

        if self.method_type == 'FL' or self.method_type == 'NV' or self.method_type == "FL_REAL":
            res.columns = pd.MultiIndex.from_product([[self.method_type],
                                                      res.columns])
        elif self.method_type == 'CM':
            res.columns = pd.MultiIndex.from_product([[f'{self.method_type}_{self.EV}'],
                                                      res.columns])
        res.index.name = 'Window'
        res = res.iloc[:-1]
        return res

    def perf_rolling_ov(self) -> pd.DataFrame:
        """
        <DESCIRPTION>
        Caculate overall performance measures for each window.
        """
        mse = self.perf_msre.perf_mse
        corr = self.perf_msre.perf_corr
        turnover = self.weights_save.diff().abs().sum(axis=1).mean() * 12

        temp = [mse, corr, turnover]
        temp = [round(item, 4) for item in temp]
        res = pd.DataFrame(temp, index=['MSE', 'Correl.', 'Turnover (Yearly)'])

        if self.method_type == 'FL' or self.method_type == 'NV' or self.method_type == 'FL_REAL':
            res.columns = [self.method_type]
        elif self.method_type == 'CM':
            res.columns = [f'{self.method_type}_{self.EV}']
        res.columns = pd.MultiIndex.from_product([['Overall OOS performance for all windows'],
                                                  res.columns])
        return res

    def perf_rolling_ov_periods(self, start_point: int, end_point: int) -> pd.DataFrame:
        """
        <DESCIRPTION>
        Caculate overall performance measures for each window.
        """
        # NOTE: BASE: FL 250, 20
        # STOCKS OVER-USED PERIODS
        # 2011-08-25 ~ 2013-04-30 (8~28, 8*20 ~ 29*20)
        # 2019-08-12 ~ 2021-03-25 (106 ~ 125, 106*20 ~ 126*20)
        # STOCKS UNDER-USED PERIODS
        # 2013-05-03 ~ 2019-08-08 (29 ~ 105, 29*20 ~ 106*20)

        def periods_slicer(start_point, end_point):
            replicated_idx = self.replicated_idx.copy()[1:]
            original_idx = self.original_idx.copy()[1:]
            replicated_idx = replicated_idx[start_point *
                                            self.freq_2: end_point*self.freq_2]
            original_idx = original_idx[start_point *
                                        self.freq_2: end_point*self.freq_2]
            return replicated_idx, original_idx

        replicated_idx, original_idx = periods_slicer(start_point, end_point)

        perf_msre_temp = PerformanceMeasure(replicated_idx, original_idx)
        mse = perf_msre_temp.perf_mse
        corr = perf_msre_temp.perf_corr
        turnover = self.weights_save[start_point:end_point].diff(
        ).abs().sum(axis=1).mean() * 12

        temp = [mse, corr, turnover]
        temp = [round(item, 4) for item in temp]
        res = pd.DataFrame(temp, index=['MSE', 'Correl.', 'Turnover (Yearly)'])

        if self.method_type == 'FL' or self.method_type == 'NV' or self.method_type == 'FL_REAL':
            res.columns = [self.method_type]
        elif self.method_type == 'CM':
            res.columns = [f'{self.method_type}_{self.EV}']
        res.columns = pd.MultiIndex.from_product([['Overall OOS performance for all windows'],
                                                  res.columns])
        return res

    # def perf_avg_mktcap(self):
    #     """
    #     <DESCIRPTION>
    #     Caculate average mktcap for each time-series.

    #     <NOTIFICATION>
    #     Not used!
    #     """
    #     start_date = self.backtest_period[0]
    #     end_date = self.backtest_period[-1]

    #     weights_save = self.weights_save.reindex(self.mktcap.index,
    #                                              method='ffill')
    #     weights_save = weights_save.notna().astype(int).replace(0, np.nan)
    #     weights_save = weights_save.loc[start_date:end_date]
    #     mktcap = self.mktcap.loc[start_date:end_date][weights_save.columns]

    #     res = weights_save * mktcap
    #     return res


def perf_trans_concat(method_type: str = 'FL'):
    """
    <DESCIRPTION>
    Caculate performance measures for each window before and after applying transaction cost.
    """
    res_dict = {}

    freq_1s = [125, 250, 375, 500]
    freq_2s = [5, 10, 15, 20]
    for val_1, val_2 in itertools.product(freq_1s, freq_2s):
        freq_1 = val_1
        freq_2 = val_2
        perf = Performance(freq_1=freq_1,
                           freq_2=freq_2,
                           method_type=method_type,
                           EV=0.99,
                           num=100)

        replicated_idx = pd.read_pickle(
            f'./RUNNER_FL_Y15_p_val_0.1/RUNNER_FL_{freq_1}_{freq_2}_p_val_0.1/replica_Y15.pkl')

        if (method_type == 'FL') or (method_type == 'FL_REAL'):
            trans_cost = perf.trans_cost_raw.multiply(
                perf.weights_save).fillna(0).sum(axis=1)[1:]
        else:
            trans_cost = perf.trans_cost_raw.multiply(
                perf.weights_save).fillna(0).sum(axis=1)

        replicated_idx = replicated_idx.values.flatten() - trans_cost.values
        replicated_idx = (1 + Func().func_plot_init_price(pd.DataFrame(replicated_idx),
                                                          100)).cumprod() - 1
        replicated_idx = replicated_idx.values.flatten()
        original_idx = perf.original_idx

        perf_msre = PerformanceMeasure(replicated_idx, original_idx)

        msres = np.array([perf_msre.perf_mean,
                          perf_msre.perf_stdev,
                          perf_msre.perf_mad,
                          perf_msre.perf_supmod,
                          perf_msre.perf_mse,
                          perf_msre.perf_corr])
        msres = np.round(msres, decimals=4)

        df = pd.DataFrame(msres, columns=['{}'.format(freq_2)], index=[
            'MEAN',
            'STDEV',
            'MAD',
            'SUPMOD',
            'MSE',
            'CORR'])

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


def perf_concat_temp(method_type: str = 'CM',
                     EV: float = 0.99,
                     num: int = 100) -> dict:
    """
    <DESCRIPTION>
    Iterate performance calculation for each methods.
    """
    res_dict = {}

    freq_1s = [125, 250, 375, 500]
    freq_2s = [5, 10, 15, 20]
    for val_1, val_2 in itertools.product(freq_1s, freq_2s):
        freq_1 = val_1
        freq_2 = val_2
        perf = Performance(freq_1=freq_1,
                           freq_2=freq_2,
                           method_type=method_type,
                           EV=EV,
                           num=num)

        df = perf.perf_run()

        if freq_1 not in res_dict:
            res_dict[freq_1] = []
        res_dict[freq_1].append(df)

    for freq, dfs in res_dict.items():
        df_concat = pd.concat(dfs, axis=1)
        res_dict[freq] = df_concat
        res_dict[freq].index.name = f"IN-SAMPLE {freq}"
        res_dict[freq].columns.name = "OUT-SAMPLE"
        if method_type == 'FL' or method_type == 'NV' or method_type == 'FL_REAL':
            res_dict[freq].columns = pd.MultiIndex.from_product(
                [[method_type], res_dict[freq].columns])
        elif method_type == 'CM':
            res_dict[freq].columns = pd.MultiIndex.from_product(
                [[f'{method_type}_{EV}'], res_dict[freq].columns])
    return res_dict


def shares_count_plot():
    """
    <DESCRIPTION>
    Plot overall number of shares.
    """
    dir_common = './RUNNER_FL_Y15_p_val_0.1/'
    replica = pd.read_pickle(
        dir_common + 'RUNNER_FL_250_5_p_val_0.1/replica_Y15.pkl')

    temp = []
    freq_1s = [125, 250, 375, 500]
    freq_2s = [5, 10, 15, 20]
    for val_1, val_2 in itertools.product(freq_1s, freq_2s):
        freq_1 = val_1
        freq_2 = val_2
        shares_count = pd.read_pickle(dir_common +
                                      f'RUNNER_FL_{freq_1}_{freq_2}_p_val_0.1/shares_count_Y15.pkl')

        def plot_match_dates(shares_count):
            dates = replica.index
            res = pd.DataFrame(np.concatenate(
                [item for item in shares_count.values for _ in range(freq_2)]))
            res = res[:dates.shape[0]]
            res.index = dates
            return res

        shares_count_matched = plot_match_dates(shares_count)
        temp.append(shares_count_matched)

    res = pd.concat(temp, axis=1)

    res_scaled = pd.DataFrame(MinMaxScaler().fit_transform(res))
    res_mean = res_scaled.mean(axis=1)
    p = Polynomial.fit(res_scaled.index, res_mean, 7)
    x = np.linspace(0, len(res_scaled), num=len(res_scaled))
    y = p(x)

    res_scaled.index = replica.index

    plt.figure(figsize=(30, 10))
    plt.title('표본 내 기간과 표본 외 기간에 따른 리더 주식 개수: FL-Adjusted')
    plt.xlabel('년도')
    plt.ylabel('종목 개수 (Min-Max 스케일링)')

    for col in res_scaled.columns:
        plt.plot(res.index, res_scaled[col], linewidth=1, linestyle='--')
    plt.plot(res_scaled.index, y, color='black', linewidth=3,
             alpha=1, label='리더 주식 개수 추세선')

    plt.fill_between(res_scaled.index, 0, 1, where=(
        res_scaled.index >= '2012-01-01') & (res_scaled.index <= '2013-01-01'),
        color='lightcoral', alpha=0.5, label='리더 주식 개수 급증 구간')
    plt.fill_between(res_scaled.index, 0, 1, where=(
        res_scaled.index >= '2020-01-01') & (res_scaled.index <= '2021-01-01'),
        color='lightcoral', alpha=0.5)
    # plt.fill_between(res_scaled.index, 0, 1, where=(
    #     res_scaled.index >= '2013-05-03') & (res_scaled.index <= '2019-08-08'),
    #     color='lightblue', alpha=0.5, label='Number of shares under-used')

    plt.legend(loc='best')
    # plt.show()
    plt.savefig('./RUNNER_GRAPHS_ETC/plot_shares_count.jpg',
                format='jpeg',
                bbox_inches='tight')


@time_spent_decorator
def perf_concat():
    """
    <DESCRIPTION>
    Concat results from perf_concat_temp.

    <NOTIFICATION>
    Call results by res_merged[freq_1].
    """
    res_merged = {}

    res_fl_real = perf_concat_temp(method_type='FL_REAL')
    res_fl = perf_concat_temp(method_type='FL')
    # res_cm = perf_concat_temp(method_type='CM',
    #                           EV=0.9)
    res_cm_99 = perf_concat_temp(method_type='CM',
                                 EV=0.99)
    res_cm_999 = perf_concat_temp(method_type='CM',
                                  EV=0.999)
    res_nv = perf_concat_temp(method_type='NV',
                              num=100)
    for key in res_fl.keys():
        res_merged[key] = pd.concat(
            [res_fl_real[key], res_fl[key],
             res_cm_99[key], res_cm_999[key],
             res_nv[key]], axis=1)
    return res_merged


@time_spent_decorator
def perf_concat_table() -> pd.DataFrame:
    """
    <DESCRIPTION>
    Concat results for every in-sample and out-sample.
    """
    res = perf_concat()
    for df in res.values():
        df.rename_axis(index=None, inplace=True)
    res_merged = pd.concat([res[125], res[250], res[375], res[500]],
                           axis=0,
                           keys=['125', '250', '375', '500'],
                           join='outer')
    return res_merged


@time_spent_decorator
def perf_concat_rolling(freq_1: int = 250,
                        freq_2: int = 20) -> pd.DataFrame:
    """
    <DESCIRPTION>
    Concat results from perf_rolling & perf_rolling_ov.
    """
    perf_fl_real = Performance(freq_1=freq_1,
                               freq_2=freq_2,
                               method_type='FL_REAL')
    perf_fl = Performance(freq_1=freq_1,
                          freq_2=freq_2,
                          method_type='FL')
    # perf_cm = Performance(freq_1=freq_1,
    #                       freq_2=freq_2,
    #                       method_type='CM',
    #                       EV=0.9)
    perf_cm_99 = Performance(freq_1=freq_1,
                             freq_2=freq_2,
                             method_type='CM',
                             EV=0.99)
    perf_cm_999 = Performance(freq_1=freq_1,
                              freq_2=freq_2,
                              method_type='CM',
                              EV=0.999)
    perf_nv = Performance(freq_1=freq_1,
                          freq_2=freq_2,
                          method_type='NV')

    res_fl_real = perf_fl_real.perf_rolling()
    res_fl = perf_fl.perf_rolling()
    # res_cm = perf_cm.perf_rolling()
    res_cm_99 = perf_cm_99.perf_rolling()
    res_cm_999 = perf_cm_999.perf_rolling()
    res_nv = perf_nv.perf_rolling()

    res_fl_real_ov = perf_fl_real.perf_rolling_ov()
    res_fl_ov = perf_fl.perf_rolling_ov()
    # res_cm_ov = perf_cm.perf_rolling_ov()
    res_cm_99_ov = perf_cm_99.perf_rolling_ov()
    res_cm_999_ov = perf_cm_999.perf_rolling_ov()
    res_nv_ov = perf_nv.perf_rolling_ov()

    res_merged = pd.concat(
        [res_fl_real, res_fl, res_cm_99, res_cm_999, res_nv], axis=1)
    res_merged.reset_index(drop=True, inplace=True)
    res_merged.index.name = 'Window'
    res_merged_ov = pd.concat(
        [res_fl_real_ov, res_fl_ov,
         res_cm_99_ov, res_cm_999_ov,
         res_nv_ov], axis=1)
    return res_merged, res_merged_ov


def perf_concat_rolling_periods(freq_1: int = 250,
                                freq_2: int = 20,
                                start_point: int = 8,
                                end_point: int = 29) -> pd.DataFrame:
    """
    <DESCIRPTION>
    Concat results from perf_rolling & perf_rolling_ov_periods between start_point and end_point.

    <NOTIFICATION>
    BASE: FL 250, 20

    STOCKS OVER-USED PERIODS
    2011-08-25 ~ 2013-04-30 (8~28, 8*20 ~ 29*20)
    2019-08-12 ~ 2021-03-25 (106 ~ 125, 106*20 ~ 126*20)

    STOCKS UNDER-USED PERIODS
    2013-05-03 ~ 2019-08-08 (29 ~ 105, 29*20 ~ 106*20)
    """
    perf_fl_real = Performance(freq_1=freq_1,
                               freq_2=freq_2,
                               method_type='FL_REAL')
    perf_fl = Performance(freq_1=freq_1,
                          freq_2=freq_2,
                          method_type='FL')
    # perf_cm = Performance(freq_1=freq_1,
    #                       freq_2=freq_2,
    #                       method_type='CM',
    #                       EV=0.9)
    perf_cm_99 = Performance(freq_1=freq_1,
                             freq_2=freq_2,
                             method_type='CM',
                             EV=0.99)
    perf_cm_999 = Performance(freq_1=freq_1,
                              freq_2=freq_2,
                              method_type='CM',
                              EV=0.999)
    perf_nv = Performance(freq_1=freq_1,
                          freq_2=freq_2,
                          method_type='NV')

    res_fl_real = perf_fl_real.perf_rolling()[start_point:end_point]
    res_fl = perf_fl.perf_rolling()[start_point:end_point]
    # res_cm = perf_cm.perf_rolling()[start_point:end_point]
    res_cm_99 = perf_cm_99.perf_rolling()[start_point:end_point]
    res_cm_999 = perf_cm_999.perf_rolling()[start_point:end_point]
    res_nv = perf_nv.perf_rolling()[start_point:end_point]

    res_fl_real_ov = perf_fl_real.perf_rolling_ov_periods(
        start_point, end_point)
    res_fl_ov = perf_fl.perf_rolling_ov_periods(start_point, end_point)
    # res_cm_ov = perf_cm.perf_rolling_ov_periods(start_point, end_point)
    res_cm_99_ov = perf_cm_99.perf_rolling_ov_periods(start_point, end_point)
    res_cm_999_ov = perf_cm_999.perf_rolling_ov_periods(start_point, end_point)
    res_nv_ov = perf_nv.perf_rolling_ov_periods(start_point, end_point)

    res_merged = pd.concat(
        [res_fl_real, res_fl,
         res_cm_99, res_cm_999,
         res_nv], axis=1)
    res_merged.reset_index(drop=True, inplace=True)
    res_merged.index.name = 'Window'
    res_merged_ov = pd.concat(
        [res_fl_real_ov, res_fl_ov,
         res_cm_99_ov, res_cm_999_ov,
         res_nv_ov], axis=1)
    return res_merged, res_merged_ov


if __name__ == "__main__":
    perf_fl = Performance(method_type='NV',
                          date='Y15',
                          freq_1=500,
                          freq_2=20,
                          EV=0.999,
                          mkt='KOSPI200',
                          start_date='2011-01-01')
    # df_real = perf_trans_concat(method_type='FL_REAL')

    # shares_count_plot()

    window_performance = perf_fl.perf_rolling_values()
    window_performance = np.round(
        window_performance[-12:].reset_index(drop=True), 4)

    perf_fl.perf_rolling_ov_periods(-13, -1)

    # res = perf_concat()
    # res_table = perf_concat_table()
    # res_rolling, res_rolling_ov = perf_concat_rolling(freq_1=250,
    #                                                   freq_2=5)

    # NOTE: BASE: FL 250, 20
    # STOCKS OVER-USED PERIODS
    # 2011-08-25 ~ 2013-04-30 (8~29, 8*20 ~ 29*20)
    # 2019-08-12 ~ 2021-03-25 (106 ~ 126, 106*20 ~ 126*20)
    # STOCKS UNDER-USED PERIODS
    # 2013-05-03 ~ 2019-08-08 (29 ~ 106, 29*20 ~ 106*20)

    # NOTE: BASE: FL 250, 5
    # STOCKS OVER-USED PERIODS
    # 2012-01-01 ~ 2012-06-30 (50 ~ 74)
    # 2020-07-01 ~ 2020-12-31 (468 ~ 492)
    # STOCKS UNDER-USED PERIODS
    # 2013-07-01 ~ 2013-12-31 (124 ~ 148)
    # 2018-07-01 ~ 2018-12-31 (369 ~ 393)

    # NOTE: BASE: FL 520, 20
    # USE -12 windows

    # start_point = -13
    # end_point = -1

    # res_rolling_periods, res_rolling_ov_periods = perf_concat_rolling_periods(freq_1=500,
    #                                                                           freq_2=20,
    #                                                                           start_point=start_point,
    #                                                                           end_point=end_point)
    # print(f'{res_rolling}\n{res_rolling_ov}', end='')
    # print("\n ***** MOVING ON: PERIOD ROLLING ***** \n")
    # print(f'{res_rolling_periods}\n{res_rolling_ov_periods}', end='')

    # trans_res = perf_trans_concat('NV')
