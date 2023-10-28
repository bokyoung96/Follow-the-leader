"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import os
import datetime
import itertools
from pathlib import Path

from empirical_weights_CM import *


# LOCATE DIRECTORY
def locate_dir(dir_name):
    path = Path(dir_name)
    path.mkdir(parents=True, exist_ok=True)


# FREQUENCY (IN, OUT)
freq_1 = 125
freq_2 = 5
date = datetime.date.today()
ev = 0.9
start_date = '2011-01-01'


# RUNNER_{freq_1}__{freq_2}_{date}_{ev}
dir_global = "RUNNER_CM_{}_{}_{}_ev_{}".format(
    freq_1, freq_2, date, ev)

locate_dir("RUNNER_GRAPHS_CM")


class DataSplit:
    def __init__(self, mkt: str = 'KOSPI200',
                 date: str = 'Y5',
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


class MethodRunnerCM(Func):
    def __init__(self,
                 EV: float = 0.9,
                 min_R2: float = 0.8,
                 mkt: str = 'KOSPI200',
                 date: str = 'Y15',
                 idx_weight: str = 'EQ'
                 ):
        """
        <DESCRIPTION>
        Run from leader stock selection to weight optimization.

        <PARAMETER>
        Same as EmMethodCM and DataLoader.

        <CONSTRUCTOR>
        Same as DataSplit.
        """
        super().__init__()
        self.EV = EV
        self.min_R2 = min_R2
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
        res = []
        count = 0
        shares_count = []
        F_nums_count = []
        weights_count = []
        weights_save = pd.DataFrame()
        success_count = 0
        sample_division = -freq_2
        self.start_date_adj = None
        for stocks in self.splits:
            # NOTE: Drop process executed in runner.
            consts = self.consts.loc[stocks.index[-1]]
            stocks = stocks.loc[:, stocks.columns.isin(
                consts[consts == 1].index)]
            stocks = stocks.dropna(how='any', axis=1)
            for col in stocks.columns:
                if stocks[col].nunique() == 1:
                    stocks = stocks.drop(col, axis=1)
            stocks = stocks.astype(np.float64)
            print("PREPROCESS DONE. MOVING ON...")

            in_sample = stocks.iloc[:sample_division]
            out_sample_ret = stocks.pct_change().iloc[sample_division:]
            if self.start_date_adj is None:
                self.start_date_adj = out_sample_ret.index[0]
            else:
                pass

            in_sample_idx = self.data_split.idx[in_sample.index]

            if stocks.shape[0] < self.years + self.months - 1:
                print("ITERATION LIMIT REACHED. FINISHING...")
                break

            while True:
                weights = EmWeightsCM(idx=in_sample_idx,
                                      stocks=in_sample,
                                      EV=self.EV,
                                      min_R2=self.min_R2)

                opt_weights, opt_res, save = weights.optimize()
                if opt_res.success:
                    print("\n***** OPTIMIZATION SUCCESS *****\n")
                    success_count += 1
                    break
                else:
                    print("\n***** OPTIMIZATION FAILED *****\n")
                    opt_weights = np.full(
                        (1, weights.shares_n), weights.shares_n)
                    break
            save.index = [out_sample_ret.index[0]]
            weights_save = pd.concat([weights_save, save], ignore_index=False)

            leaders_out_sample = np.array(
                out_sample_ret.T.iloc[weights.get_matched_rows(), :])
            out_sample_res = np.dot(opt_weights, leaders_out_sample)

            res.append(out_sample_res)
            shares_count.append(weights.shares_n)
            F_nums_count.append(weights.F_nums)
            weights_count.append(opt_weights)

            count += 1
            print("ATTEMPT {} OF {} COMPLETED. MOVING ON...".format(
                count, len(self.splits)))
            # pd.DataFrame(out_sample_res).to_pickle(
            #     "./{}/replica_{}_{}.pkl".format(dir_global, count, self.date))
            pd.DataFrame(weights.get_matched_rows()).to_pickle(
                "./{}/replica_matched_{}_{}.pkl".format(dir_global, count, self.date))

        pd.DataFrame(shares_count).to_pickle(
            "./{}/shares_count_{}.pkl".format(dir_global, self.date))
        pd.DataFrame(F_nums_count).to_pickle(
            "./{}/F_nums_count_{}.pkl".format(dir_global, self.date))
        pd.DataFrame(weights_save).to_pickle(
            "./{}/weights_save_{}.pkl".format(dir_global, self.date))

        res_df = pd.DataFrame(np.concatenate(res).flatten())
        res_df.to_pickle(
            "./{}/replica_{}.pkl".format(dir_global, self.date))

        sums = []
        for arr in weights_count:
            arr_sum = np.sum(arr)
            sums.append(arr_sum)
        weights_count_df = pd.DataFrame(sums)
        weights_count_df.to_pickle(
            "./{}/weights_count_{}.pkl".format(dir_global, self.date))
        return res_df, shares_count

    def runner_plot(self, init_price: int = 100) -> plt.plot:
        """
        <DESCRIPTION>
        Plot out-sample results.
        """
        replica = pd.read_pickle(
            "./{}/replica_{}.pkl".format(dir_global, self.date))
        idx_ret = DataSplit(self.mkt, self.date,
                            self.idx_weight).idx.pct_change()
        idx_ret = idx_ret[self.start_date_adj:]
        original = idx_ret[:len(replica)]

        replica = (1 + self.func_plot_init_price(replica,
                   init_price)).cumprod() - 1
        original = (1 + self.func_plot_init_price(original,
                    init_price)).cumprod() - 1

        original.rename(
            index={0: original.index[1] - pd.DateOffset(days=1)}, inplace=True)

        shares_count = pd.read_pickle(
            "./{}/shares_count_{}.pkl".format(dir_global, self.date))
        shares_count = pd.DataFrame(np.concatenate(
            [item for item in shares_count.values for _ in range(freq_2)]))
        shares_count = self.func_plot_init_price(shares_count, np.nan)
        shares_count_mean = int(shares_count[1:].mean().values)

        replica.index = original.index
        shares_count.index = original.index

        shares_count_diff = shares_count.diff()
        slope_change_idx = shares_count_diff.index[shares_count_diff[0] != 0]
        markevery = [shares_count.index.get_loc(i) for i in slope_change_idx]

        fig, ax1 = plt.subplots(figsize=(25, 10))
        plt.title('REPLICA VERSUS ORIGINAL: {}, {}, {}'.format(
            self.date, freq_1, freq_2))

        ax1.plot(replica, label='REPLICA', color='r', linewidth=2.5)
        ax1.plot(original, label='ORIGINAL', color='black', linewidth=2.5)
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

        # for i, index in enumerate(markevery[1:]):
        #     value = shares_count.iloc[index][0]
        #     index_date = shares_count.index[index]
        #     ax2.annotate(f"{int(value)}",
        #                  (index_date, value),
        #                  textcoords="offset points",
        #                  xytext=(0, -20),
        #                  ha='center',
        #                  fontsize=10,
        #                  color='black',
        #                  rotation=45)
        plt.savefig(
            './RUNNER_GRAPHS_CM/{}.jpg'.format(dir_global), format='jpeg')
        # plt.show()


if __name__ == "__main__":
    freq_1s = [125, 250, 375]
    freq_2s = [20]
    for val_1, val_2 in itertools.product(freq_1s, freq_2s):
        freq_1 = val_1
        freq_2 = val_2
        dir_global = "RUNNER_CM_{}_{}_{}_ev_{}".format(
            freq_1, freq_2, date, ev)
        locate_dir("./{}/".format(dir_global))

        runner = MethodRunnerCM(date='Y15')
        if os.path.exists('./{}/'.format(dir_global)):
            res_df, shares_count = runner.runner()
            runner.runner_plot(init_price=100)
            print("DONE. MOVING ON...")

        # runner = MethodRunnerCM(date='Y15')
        # res_df, shares_count, weights_count = runner.runner()
        # runner.runner_plot(init_price=100)
