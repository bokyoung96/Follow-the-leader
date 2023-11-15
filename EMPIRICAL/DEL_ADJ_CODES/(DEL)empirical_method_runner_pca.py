"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import os
import itertools
from pathlib import Path

from empirical_func import *
from empirical_loader import *

# LOCATE DIRECTORY


def locate_dir(dir_name):
    path = Path(dir_name)
    path.mkdir(parents=True, exist_ok=True)


# FREQUENCY (IN, OUT)
freq_1 = 250
freq_2 = 5
p_val = 0.1
start_date = '2011-01-01'


class DataSplit:
    def __init__(self, mkt: str = 'KOSPI200',
                 date: str = 'Y15',
                 idx_weight: str = 'EQ'):
        """
        <DESCRIPTION>
        Split datas for in-sample and out-sample tests.

        <PARAMETER>
        mkt: Market specified for data.
        date: Date specified for data.
        idx_weight: Index weight specified for data.

        <CONSTRUCTOR>
        idx, stocks: Index and stock data.
        months: Month frequency by global var freq_2.
        years: Year frequency by global var freq_1.
        """
        self.data_loader = DataLoader(mkt, date)
        self.idx, self.stocks = self.data_loader.as_empirical(
            idx_weight=idx_weight)
        self.months = freq_2
        self.years = freq_1

    def data_split(self):
        """
        <DESCRIPTION>
        Split datas into in-sample + out-sample.
        """
        res = []
        start = 0

        temp_start = self.stocks.loc[start_date:]
        temp_window = self.stocks.loc[:start_date].iloc[-self.years:]
        self.stocks = pd.concat([temp_window, temp_start], axis=0)
        rows = len(self.stocks)
        while start < rows:
            end = start + self.years + self.months

            if end > rows:
                end = rows
            temp = self.stocks.iloc[start:end]
            res.append(temp)
            start = start + self.months
        print("SPLIT COMPLETED. MOVING ON...")
        return res


class EmMethodPCA(Func):
    def __init__(self,
                 idx: pd.Series,
                 stocks: pd.DataFrame,
                 F_max: int = 30,
                 EV: float = 0.90,
                 ):
        super().__init__()
        self.idx = idx
        self.stocks = stocks.astype(np.float64)
        self.F_max = F_max
        self.EV = EV

        self.stocks_ret = self.stocks.copy()
        self.stocks_ret = self.stocks_ret.pct_change(axis=0).iloc[1:, :]

        self.idx_ret = self.idx.copy()
        self.idx_ret = self.idx_ret.pct_change()[1:]

        self.get_factors

    @property
    def get_factors(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get number of factors under Bai and Ng.
        Get factors and factor loadings under PCA.
        """
        self.F_nums = self.func_bai_ng(
            self.stocks_ret, ic_method=2, max_factor=self.F_max)[0]

        pca, PC = self.func_pca(n_components=self.F_max,
                                df=self.stocks_ret)

        self.explained_variance = pca.explained_variance_ratio_
        self.explained_variance_cumsum = np.cumsum(
            pca.explained_variance_ratio_)


class MethodRunnerPCA(Func):
    def __init__(self,
                 F_max: int = 30,
                 EV: float = 0.9,
                 mkt: str = 'KOSPI200',
                 date: str = 'Y15',
                 idx_weight: str = 'EQ'
                 ):
        """
        <DESCRIPTION<
        Run from leader stock selection to weight optimization.

        <PARAMETER>
        Same as EmMethodFL and DataLoader.

        <CONSTRUCTOR>
        Same as DataSplit.
        """
        super().__init__()
        self.F_max = F_max
        self.EV = EV
        self.mkt = mkt
        self.date = date
        self.idx_weight = idx_weight
        self.months = freq_2
        self.years = freq_1

        self.data_split = DataSplit(self.mkt,
                                    self.date,
                                    self.idx_weight)
        self.splits = self.data_split.data_split()

        self.consts = pd.read_pickle(
            f"./{self.mkt}_CONST/{self.mkt}_CONST_{self.date}_PIVOT.pkl")
        print("CONSTITUENTS LOADED. MOVING ON...")

    @time_spent_decorator
    def runner(self):
        """
        <DESCRIPTION>
        Select leader stocks and optimize its weights under in-sample and apply into out-sample.
        """
        pca_ev = []
        pca_ev_cumsum = []
        count = 0
        sample_division = -freq_2
        self.start_date_adj = None
        for stocks in self.splits:
            consts = self.consts.loc[stocks.index[-1]]
            stocks = stocks.loc[:, stocks.columns.isin(
                consts[consts == 1].index)]
            stocks = stocks.dropna(how='any', axis=1)
            for col in stocks.columns:
                if stocks[col].nunique() == 1:
                    stocks = stocks.drop(col, axis=1)
            print("PREPROCESS DONE. MOVING ON...")

            in_sample = stocks.iloc[:sample_division]
            out_sample_ret = stocks.pct_change().iloc[sample_division-1:-1]
            if self.start_date_adj is None:
                self.start_date_adj = out_sample_ret.index[0]
            else:
                pass

            in_sample_idx = self.data_split.idx[in_sample.index]

            if stocks.shape[0] < self.years + self.months - 1:
                print("ITERATION LIMIT REACHED. FINISHING...")
                break

            while True:
                res = EmMethodPCA(idx=in_sample_idx,
                                  stocks=in_sample,
                                  F_max=self.F_max,
                                  EV=self.EV)

                ev = res.explained_variance
                ev_cumsum = res.explained_variance_cumsum

                pca_ev.append(ev)
                pca_ev_cumsum.append(ev_cumsum)
                break
            count += 1
            print(f"ATTEMPT {count} ACCOMPLISHED. MOVING ON...")
        return pca_ev, pca_ev_cumsum


if __name__ == "__main__":
    freq_1s = [250]
    freq_2s = [5]
    for val_1, val_2 in itertools.product(freq_1s, freq_2s):
        freq_1 = val_1
        freq_2 = val_2

        runner = MethodRunnerPCA(date='Y15')
        pca_ev, pca_ev_cumsum = runner.runner()
