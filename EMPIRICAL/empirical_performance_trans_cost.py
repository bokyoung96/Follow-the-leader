"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from empirical_func import *
from empirical_loader import *

# LOCATE DIRECTORY


def locate_dir(dir_name):
    path = Path(dir_name)
    path.mkdir(parents=True, exist_ok=True)


locate_dir('RUNNER_GRAPH_TRS')

start_date = '2011-01-01'
end_date = '2022-12-31'


class PerformanceTrs(DataLoader):
    stopper_idx = None

    def __init__(self,
                 method_type: str = 'CM',
                 date: str = 'Y15',
                 freq_1: int = 250,
                 freq_2: int = 20,
                 EV: float = 0.9,
                 num: int = 100,
                 rolls: int = 9,
                 mkt: str = 'KOSPI200',
                 idx_weight: str = 'EQ'):
        """
        <CAUTION>
        USE AFTER DateMatcher IS DONE FOR PICKLE FILES.
        """
        super().__init__(mkt, date)
        self.method_type = method_type
        self.date = date
        self.freq_1 = freq_1
        self.freq_2 = freq_2
        self.EV = EV
        self.num = num
        self.rolls = rolls

        if PerformanceTrs.stopper_idx is None:
            PerformanceTrs.stopper_idx = self.as_empirical(
                idx_weight=idx_weight)[0]

        self.idx = PerformanceTrs.stopper_idx.pct_change(
        ).loc[start_date:end_date]

        if self.method_type == 'FL':
            self.dir_global = f"RUNNER_{self.method_type}_{self.date}_p_val_0.1"
            self.dir_global_trs = f"RUNNER_{self.method_type}_{self.date}_p_val_0.1_TRS"
            self.dir_global_adj = f"RUNNER_{self.method_type}_{self.date}_p_val_0.1_ADJ"
            self.dir_main = f"RUNNER_{self.method_type}_{self.freq_1}_{self.freq_2}_p_val_0.1"
            self.dir_main_trs = f"RUNNER_{self.method_type}_{self.freq_1}_{self.freq_2}_p_val_0.1_{self.rolls}"
            self.dir_main_adj = f"RUNNER_{self.method_type}_{self.freq_1}_{self.freq_2}_p_val_0.1_adj"

        elif self.method_type == "CM":
            self.dir_global = f"RUNNER_{self.method_type}_{self.date}_ev_{self.EV}"
            self.dir_main = f"RUNNER_{self.method_type}_{self.freq_1}_{self.freq_2}_ev_{self.EV}"

        elif self.method_type == "NV":
            self.dir_global = f"RUNNER_{self.method_type}_{self.date}_num_{self.num}"
            self.dir_main = f"RUNNER_{self.method_type}_{self.freq_1}_{self.freq_2}_num_{self.num}"

        else:
            raise AssertionError("Invalid Value. Insert FL / CM / NV.")

        self.weights_save = pd.read_pickle(
            f'./{self.dir_global}/{self.dir_main}/weights_save_{self.date}.pkl')
        self.weights_save = self.weights_save.loc[start_date:end_date]

        self.trans_cost_raw = pd.read_pickle(
            f'./{mkt}_TRANSACTION_COST/{mkt}_TRANSACTION_COST_{self.date}.pkl')
        self.trans_cost_raw = self.trans_cost_raw.loc[start_date:end_date]
        self.trans_cost_raw = self.trans_cost_raw[self.weights_save.columns]

        self.replica_raw = pd.read_pickle(
            f'./{self.dir_global}/{self.dir_main}/replica_{self.date}.pkl')
        self.replica_raw = self.replica_raw[0]

        self.shares_count = pd.read_pickle(
            f'./{self.dir_global}/{self.dir_main}/shares_count_{self.date}.pkl')

        if self.method_type == 'FL':
            # TRS
            self.weights_save_trs = pd.read_pickle(
                f'./{self.dir_global_trs}/{self.dir_main_trs}/weights_save_{self.date}.pkl')
            self.weights_save_trs = self.weights_save_trs.loc[start_date:end_date]

            self.trans_cost_raw_trs = pd.read_pickle(
                f'./{mkt}_TRANSACTION_COST/{mkt}_TRANSACTION_COST_{self.date}.pkl')
            self.trans_cost_raw_trs = self.trans_cost_raw_trs.loc[start_date:end_date]
            self.trans_cost_raw_trs = self.trans_cost_raw_trs[self.weights_save_trs.columns]

            self.replica_raw_trs = pd.read_pickle(
                f'./{self.dir_global_trs}/{self.dir_main_trs}/replica_{self.date}.pkl')
            self.replica_raw_trs = self.replica_raw_trs[0]

            self.shares_count_trs = pd.read_pickle(
                f'./{self.dir_global_trs}/{self.dir_main_trs}/shares_count_{self.date}.pkl')

            # ADJ
            self.weights_save_adj = pd.read_pickle(
                f'./{self.dir_global_adj}/{self.dir_main_adj}/weights_save_{self.date}.pkl')
            self.weights_save_adj = self.weights_save_adj.loc[start_date:end_date]

            self.trans_cost_raw_adj = pd.read_pickle(
                f'./{mkt}_TRANSACTION_COST/{mkt}_TRANSACTION_COST_{self.date}.pkl')
            self.trans_cost_raw_adj = self.trans_cost_raw_adj.loc[start_date:end_date]
            self.trans_cost_raw_adj = self.trans_cost_raw_adj[self.weights_save_adj.columns]

            self.replica_raw_adj = pd.read_pickle(
                f'./{self.dir_global_adj}/{self.dir_main_adj}/replica_{self.date}.pkl')
            self.replica_raw_adj = self.replica_raw_adj[0]

            self.shares_count_adj = pd.read_pickle(
                f'./{self.dir_global_adj}/{self.dir_main_adj}/shares_count_{self.date}.pkl')

        else:
            pass

    @property
    def trans_cost(self):
        return self.trans_cost_raw.multiply(self.weights_save).fillna(0).sum(axis=1)

    @property
    def replica(self):
        return self.replica_raw - self.trans_cost

    @property
    def data_to_be_used(self):
        replica_raw = (1 + Func().func_plot_init_price(self.replica_raw,
                                                       100)).cumprod() - 1
        replica = (1 + Func().func_plot_init_price(self.replica,
                                                   100)).cumprod() - 1
        idx = (1 + Func().func_plot_init_price(self.idx,
                                               100)).cumprod() - 1

        replica_raw.rename(
            index={0: replica_raw.index[1] - pd.DateOffset(days=1)}, inplace=True)
        replica.rename(
            index={0: replica.index[1] - pd.DateOffset(days=1)}, inplace=True)
        idx.rename(
            index={0: idx.index[1] - pd.DateOffset(days=1)}, inplace=True)

        return replica_raw, replica, idx

    @property
    def trans_cost_trs(self):
        return self.trans_cost_raw_trs.multiply(self.weights_save_trs).fillna(0).sum(axis=1)

    @property
    def replica_trs(self):
        return self.replica_raw_trs - self.trans_cost_trs

    @property
    def data_to_be_used_trs(self):
        replica_raw_trs = (1 + Func().func_plot_init_price(self.replica_raw_trs,
                                                           100)).cumprod() - 1
        replica_trs = (1 + Func().func_plot_init_price(self.replica_trs,
                                                       100)).cumprod() - 1

        replica_raw_trs.rename(
            index={0: replica_raw_trs.index[1] - pd.DateOffset(days=1)}, inplace=True)
        replica_trs.rename(
            index={0: replica_trs.index[1] - pd.DateOffset(days=1)}, inplace=True)
        return replica_raw_trs, replica_trs

    @property
    def trans_cost_adj(self):
        return self.trans_cost_raw_adj.multiply(self.weights_save_adj).fillna(0).sum(axis=1)

    @property
    def replica_adj(self):
        return self.replica_raw_adj - self.trans_cost_adj

    @property
    def data_to_be_used_adj(self):
        replica_raw_adj = (1 + Func().func_plot_init_price(self.replica_raw_adj,
                                                           100)).cumprod() - 1
        replica_adj = (1 + Func().func_plot_init_price(self.replica_adj,
                                                       100)).cumprod() - 1

        replica_raw_adj.rename(
            index={0: replica_raw_adj.index[1] - pd.DateOffset(days=1)}, inplace=True)
        replica_adj.rename(
            index={0: replica_adj.index[1] - pd.DateOffset(days=1)}, inplace=True)
        return replica_raw_adj, replica_adj

    def plot_data(self):
        replica_raw, replica, idx = self.data_to_be_used

        plt.figure(figsize=(25, 10))
        plt.xlabel('Date')
        plt.ylabel('Cumulative return')
        plt.title('REPLICA WITH TRANSACTION COST')

        plt.plot(replica_raw, label='REPLICA', color='b', linewidth=1)
        plt.plot(replica, label='REPLICA W TRANSACTION COST',
                 color='r', linewidth=1)
        plt.plot(idx, label='ORIGINAL', color='black', linewidth=2)
        plt.fill_between(x=replica_raw.index, y1=replica_raw,
                         y2=replica, color='grey', alpha=0.3)
        plt.legend(loc='best')
        plt.show()
        # plt.savefig(f'./RUNNER_GRAPH_TRS/{self.dir_main}', format='jpeg')

    def plot_data_adj(self):
        replica_raw_adj, replica_adj = self.data_to_be_used_adj
        idx = (1 + Func().func_plot_init_price(self.idx,
                                               100)).cumprod() - 1
        idx.rename(
            index={0: idx.index[1] - pd.DateOffset(days=1)}, inplace=True)

        plt.figure(figsize=(25, 10))
        plt.xlabel('Date')
        plt.ylabel('Cumulative return')
        plt.title('REPLICA ADJ WITH TRANSACTION COST')

        plt.plot(replica_raw_adj, label='REPLICA ADJ', color='b', linewidth=1)
        plt.plot(replica_adj, label='REPLICA ADJ W TRANSACTION COST',
                 color='r', linewidth=1)
        plt.plot(idx, label='ORIGINAL', color='black', linewidth=2)
        plt.fill_between(x=replica_raw_adj.index, y1=replica_raw_adj,
                         y2=replica_adj, color='grey', alpha=0.3)
        plt.legend(loc='best')
        plt.show()

    def plot_data_difference(self):
        replica_raw, replica, idx = self.data_to_be_used
        difference = replica_raw - replica
        x = replica_raw.index

        if self.method_type == 'FL':
            replica_raw_trs, replica_trs = self.data_to_be_used_trs
            difference_trs = replica_raw_trs - replica_trs

            replica_raw_adj, replica_adj = self.data_to_be_used_adj
            difference_adj = replica_raw_adj - replica_adj

        fig, ax1 = plt.subplots(figsize=(25, 10))
        plt.title('REPLICA WITH TRANSACTION COST')

        ax1.plot(x, replica_raw, 'b-', label='REPLICA')
        ax1.plot(x, replica, 'r-', label='REPLICA W TRANSACTION COST')
        ax1.plot(x, idx, 'k-', label='ORIGINAL', linewidth=2)

        if self.method_type == 'FL':
            ax1.plot(x, replica_raw_trs, color='green',
                     label='REPLICA W ADJUSTMENT')
            ax1.plot(x, replica_trs, color='orange',
                     label='REPLICA W ADJUSTMENT W TRANSACTION COST')

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative return')

        ax2 = ax1.twinx()

        ax2.fill_between(x, 0, difference, color='grey',
                         alpha=0.5, label='CUM TRS')

        if self.method_type == 'FL':
            ax2.fill_between(x, 0, difference_trs, color='purple',
                             alpha=0.1, label='CUM TRS W ADJUSTMENT')
            ax2.fill_between(x, 0, difference_adj, color='blue',
                             alpha=0.1, label='CUM ADJ')

        ax2.set_ylabel('Transaction cost')

        fig.tight_layout()
        ax1.legend(loc='lower left')
        ax2.legend(loc='lower right')
        plt.show()
        # plt.savefig(f'./RUNNER_GRAPH_TRS/{self.dir_main}', format='jpeg')

    def table_data_difference(self):
        diff_year = self.trans_cost.resample('A').mean()
        diff_month = self.trans_cost.groupby(
            self.trans_cost.index.month).mean()

        if self.method_type == 'FL':
            diff_year_trs = self.trans_cost_trs.resample('A').mean()
            diff_month_trs = self.trans_cost_trs.groupby(
                self.trans_cost_trs.index.month).mean()

        res = pd.DataFrame({'Year': diff_year.index,
                            'Transaction cost (Y, sum)': diff_year.values,
                            'Month': diff_month.index,
                            'Transaction cost (M, mean)': diff_month.values})

        if self.method_type == 'FL':
            diff_year_btw = diff_year.values - diff_year_trs.values
            diff_month_btw = diff_month.values - diff_month_trs.values

            res = pd.DataFrame({'Year': diff_year.index,
                                'Transaction cost (Y, sum)': diff_year.values,
                                'Transaction cost (TRS, Y, sum)': diff_year_trs.values,
                                'Difference (Y, sum)': diff_year_btw,
                                'Month': diff_month.index,
                                'Transaction cost (M, mean)': diff_month.values,
                                'Transaction cost (TRS, M, mean)': diff_month_trs.values,
                                'Difference (M, mean)': diff_month_btw})
            res.index.name = "Transaction costs in (%)"

        res = np.round(res, 6)
        return res

    def table_turnover_difference(self):
        turnover = self.weights_save.diff().abs().sum(axis=1).mean()
        turnover_trs = self.weights_save_trs.diff().abs().sum(axis=1).mean()

        res = pd.DataFrame([turnover, turnover_trs],
                           index=['REPLICA',
                                  'REPLICA W ADJUSTMENT'],
                           columns=['TURNOVER']).T
        res.index.name = 'PORTFOLIO TURNOVER'
        res = np.round(res, 6)
        return res

    def plot_weight_counts(self):
        def plot_match_dates(shares_count):
            dates = self.replica.index
            res = pd.DataFrame(np.concatenate(
                [item for item in shares_count.values for _ in range(self.freq_2)]))
            res = res[:dates.shape[0]]
            res.index = dates
            return res

        def plot_markevery(shares_count):
            shares_count_diff = shares_count.diff()
            slope_change_idx = shares_count_diff.index[shares_count_diff[0] != 0]
            markevery = [shares_count.index.get_loc(
                i) for i in slope_change_idx]
            return markevery

        shares_count_matched = plot_match_dates(self.shares_count)
        shares_count_trs_matched = plot_match_dates(self.shares_count_trs)
        # shares_count_adj_matched = plot_match_dates(self.shares_count_adj)

        shares_count = plot_markevery(shares_count_matched)
        shares_count_trs = plot_markevery(shares_count_trs_matched)
        # shares_count_adj = plot_markevery(shares_count_adj_matched)

        chgs = [chg for chg in range(
            0, shares_count_trs_matched.shape[0], self.rolls * self.freq_2)]
        chgs_point = shares_count_trs_matched.iloc[chgs, :]
        chgs_matched = pd.DataFrame(np.concatenate(
            [item for item in chgs_point.values for _ in range(self.rolls * self.freq_2)]))
        chgs_matched = chgs_matched[:self.replica.index.shape[0]]
        chgs_matched.index = self.replica.index

        plt.figure(figsize=(25, 10))
        plt.title('NUMBER OF SHARES USED')
        plt.xlabel('Date')
        plt.ylabel('Number of shares')
        plt.ylim(0, 200)

        plt.plot(shares_count_matched, label='SHARES COUNT',
                 color='black', linewidth=2, linestyle='-',
                 marker='o', markevery=shares_count)
        plt.plot(shares_count_trs_matched, label='SHARES COUNT (TRS)',
                 color='b', linewidth=1.5, linestyle='--',
                 marker='o', markevery=shares_count_trs)
        plt.plot(chgs_matched, label='CHGS (TRS)',
                 color='r', linewidth=1, linestyle='--',
                 marker='x', markevery=chgs)
        # plt.plot(shares_count_adj_matched, label='SHARES COUNT (ADJ)',
        #          color='r', linewidth=1.5, linestyle='--',
        #          marker='o', markevery=shares_count_adj)

        plt.legend(loc='best')
        plt.show()


if __name__ == "__main__":
    perf_trs = PerformanceTrs(method_type='FL',
                              freq_1=250,
                              freq_2=20,
                              EV=0.99,
                              rolls=12)
    perf_trs.plot_data()
    perf_trs.plot_data_adj()
    perf_trs.plot_data_difference()
    perf_trs.plot_weight_counts()
    print(perf_trs.table_data_difference())
    print(perf_trs.table_turnover_difference())
