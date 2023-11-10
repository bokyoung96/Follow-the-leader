"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from empirical_func import *
from empirical_loader import *


start_date = '2011-01-01'
end_date = '2022-12-31'


class DateMatcher(DataLoader):
    stopper = None
    stopper_idx = None

    def __init__(self,
                 method_type: str = 'CM',
                 date: str = 'Y15',
                 freq_1: int = 250,
                 freq_2: int = 20,
                 EV: float = 0.9,
                 num: int = 100,
                 mkt: str = 'KOSPI200',
                 idx_weight: str = 'EQ'):
        """
        <NOTIFICATION>
        Match each datas into start_date and end_date.

        <CAUTION>
        ONE TIME LOADED PYTHON CODE.
        DO NOT USE IT AGAIN!
        """
        super().__init__(mkt, date)
        self.method_type = method_type
        self.date = date
        self.freq_1 = freq_1
        self.freq_2 = freq_2
        self.EV = EV
        self.num = num

        if DateMatcher.stopper is None:
            DateMatcher.stopper_idx, DateMatcher.stopper = self.as_empirical(
                idx_weight=idx_weight)

        self.idx = DateMatcher.stopper_idx.pct_change().loc[start_date:]

        if self.method_type == 'FL':
            self.dir_global = f"RUNNER_{self.method_type}_{self.date}_p_val_0.1"
            self.dir_main = f"RUNNER_{self.method_type}_{self.freq_1}_{self.freq_2}_p_val_0.1"

        elif self.method_type == "CM":
            self.dir_global = f"RUNNER_{self.method_type}_{self.date}_ev_{self.EV}"
            self.dir_main = f"RUNNER_{self.method_type}_{self.freq_1}_{self.freq_2}_ev_{self.EV}"

        elif self.method_type == "NV":
            self.dir_global = f"RUNNER_{self.method_type}_{self.date}_num_{self.num}"
            self.dir_main = f"RUNNER_{self.method_type}_{self.freq_1}_{self.freq_2}_num_{self.num}"

        else:
            raise AssertionError("Invalid Value. Insert FL / CM / NV.")

        self.get_return_data

    @property
    def get_return_data(self):
        """
        <DESCRIPTION>
        Get return datas matched.
        """
        self.replica = pd.read_pickle(
            f"./{self.dir_global}/{self.dir_main}/replica_{self.date}.pkl")
        # 1 DAY OUT-SAMPLE LAG (FL)
        if self.method_type == 'FL':
            self.replica = self.replica[1:]

        self.idx = self.idx[:self.replica.shape[0]]
        self.replica.index = self.idx.index

        self.replica = self.replica[:end_date]
        self.idx = self.idx[:end_date]

    def get_etc_data(self):
        """
        <DESCRIPTION>
        Get etc datas.
        """
        attributes = {
            "shares_count": "shares_count",
            "weights_count": "weights_count",
            "weights_save": "weights_save",
            "F_nums_count": "F_nums_count"
        }

        etc_data = {}
        for attribute, file_name in attributes.items():
            file_path = f"./{self.dir_global}/{self.dir_main}/{file_name}_{self.date}.pkl"

            if not os.path.exists(file_path):
                if attribute == "F_nums_count":
                    print("NV: NO FACTORS. MOVING ON...")
                    continue
                else:
                    raise FileNotFoundError(
                        f"The required file {file_path} does not exist.")
            data = pd.read_pickle(file_path)
            setattr(self, attribute, data)
            etc_data[attribute] = data
        return etc_data

    def pp_etc_data(self):
        """
        <DESCRIPTION>
        Preprocess etc datas.
        """
        etc_data = self.get_etc_data()
        if self.freq_2 == 5:
            x = 631 - 40
        elif self.freq_2 == 10:
            x = 315 - 19
        elif self.freq_2 == 15:
            x = 210 - 12
        elif self.freq_2 == 20:
            x = 157 - 9

        if 'F_nums_count' in etc_data:
            etc_data['F_nums_count'] = etc_data['F_nums_count'].loc[:x]
        else:
            print("NV: NO FACTORS. MOVING ON...")

        etc_data['shares_count'] = etc_data['shares_count'].loc[:x]
        etc_data['weights_save'] = etc_data['weights_save'].loc[:'2022-12-28']
        return etc_data

    def plot_data(self):
        """
        <DESCRIPTION>
        Plot return datas to be saved.
        """
        etc_data = self.pp_etc_data()
        replica = (
            1 + Func().func_plot_init_price(self.replica,
                                            100)).cumprod() - 1
        original = (1 + Func().func_plot_init_price(self.idx,
                                                    100)).cumprod() - 1
        original.rename(
            index={0: original.index[1] - pd.DateOffset(days=1)}, inplace=True)

        shares_count = etc_data['shares_count']
        shares_count = pd.DataFrame(np.concatenate(
            [item for item in shares_count.values for _ in range(self.freq_2)]))
        shares_count = shares_count[:self.replica.shape[0]]
        shares_count = Func().func_plot_init_price(shares_count, np.nan)
        shares_count_mean = int(shares_count[1:].mean().values)

        replica.index = original.index
        shares_count.index = original.index

        shares_count_diff = shares_count.diff()
        slope_change_idx = shares_count_diff.index[shares_count_diff[0] != 0]
        markevery = [shares_count.index.get_loc(i) for i in slope_change_idx]

        fig, ax1 = plt.subplots(figsize=(25, 10))
        plt.title('REPLICA VERSUS ORIGINAL: {}, {}, {}'.format(
            self.date, self.freq_1, self.freq_2))

        ax1.plot(replica, label='REPLICA', color='r', linewidth=1.5)
        ax1.plot(original, label='ORIGINAL', color='black', linewidth=1.5)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative return')
        ax1.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(shares_count, label='SHARES COUNT',
                 color='b', linewidth=1.5, linestyle='--',
                 marker='o', markevery=markevery)
        ax2.set_ylabel('Number of shares')
        ax2.set_ylim(0, 200)
        ax2.axhline(shares_count_mean, color='g',
                    linestyle='--', label='MEAN SHARES COUNT: {}'.format(shares_count_mean))
        ax2.legend(loc='lower right')

        plt.savefig(
            f'./{self.dir_global}/RUNNER_GRAPHS_{self.method_type}/{self.dir_main}.jpg', format='jpeg')
        # plt.show()

    def save_data(self):
        """
        <DESCRIPTION>
        Save all datas.
        """
        etc_data = self.pp_etc_data()

        if 'F_nums_count' in etc_data and etc_data['F_nums_count'] is not None:
            etc_data['F_nums_count'].to_pickle(
                f"./{self.dir_global}/{self.dir_main}/F_nums_count_{self.date}.pkl")
        else:
            print("NOT SAVING FACTORS. MOVING ON...")

        self.replica.to_pickle(
            f"./{self.dir_global}/{self.dir_main}/replica_{self.date}.pkl")
        etc_data['shares_count'].to_pickle(
            f"./{self.dir_global}/{self.dir_main}/shares_count_{self.date}.pkl")
        etc_data['weights_save'].to_pickle(
            f"./{self.dir_global}/{self.dir_main}/weights_save_{self.date}.pkl")
        etc_data['weights_save'].sum(axis=1).to_pickle(
            f"./{self.dir_global}/{self.dir_main}/weights_count_{self.date}.pkl")

    def save_weight_data(self):
        """
        <DESCRIPTION>
        Save weight datas.

        <NOTIFICAION>
        NOT USED!
        """
        weight_data = pd.read_pickle(
            f"./{self.dir_global}/{self.dir_main}/weights_save_{self.date}.pkl")
        weight_data.sum(axis=1).to_pickle(
            f"./{self.dir_global}/{self.dir_main}/weights_count_{self.date}.pkl")


if __name__ == "__main__":
    # method_types = ['CM', 'FL', 'NV']
    # method_types = ['FL']
    # freq_1s = [375]
    # freq_2s = [5]
    # count = 0
    # for val_1, val_2, val_3 in itertools.product(method_types, freq_1s, freq_2s):
    #     date_matcher = DateMatcher(method_type=val_1,
    #                                freq_1=val_2,
    #                                freq_2=val_3,
    #                                EV=0.999)
    #     date_matcher.plot_data()
    #     date_matcher.save_data()
    #     # date_matcher.save_weight_data()
    #     count += 1
    #     print(f"ITERATION {count} COMPLETE. MOVING ON...")

    # date_matcher = DateMatcher(method_type='CM',
    #                            freq_1=250,
    #                            freq_2=20)
    pass
