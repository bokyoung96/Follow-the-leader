"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import os
from pathlib import Path

from empirical_weights_FL import *


# LOCATE DIRECTORY
def locate_dir(dir_name):
    path = Path(dir_name)
    path.mkdir(parents=True, exist_ok=True)


# FREQUENCY (IN, OUT)
freq_1 = 375
freq_2 = 20

# RUNNER_{freq_1}__{freq_2}_RUNDATE
dir_global = "RUNNER_{}_{}_1021".format(freq_1, freq_2)
locate_dir("./{}/".format(dir_global))


class DataSplit:
    def __init__(self, mkt: str = 'KOSPI200',
                 date: str = 'Y3',
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


class MethodRunner:
    def __init__(self,
                 F_max: int = 30,
                 EV: float = 0.9,
                 mkt: str = 'KOSPI200',
                 date: str = 'Y3',
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

    @time_spent_decorator
    def runner(self):
        """
        <DESCRIPTION>
        Select leader stocks and optimize its weights under in-sample and apply into out-sample.
        """
        res = []
        count = 0
        shares_count = []
        success_count = 0
        sample_division = -freq_2
        for stocks in self.splits:
            in_sample = stocks.iloc[:sample_division]
            out_sample_ret = stocks.pct_change().iloc[sample_division-1:-1]

            if stocks.shape[0] < self.years + self.months - 1:
                print("ITERATION LIMIT REACHED. FINISHING...")
                break

            while True:
                weights = EmWeightsFL(idx=self.data_split.idx,
                                      stocks=in_sample,
                                      F_max=self.F_max,
                                      EV=self.EV)

                opt_weights, opt_res = weights.optimize()
                if opt_res.success:
                    print("\n***** OPTIMIZATION SUCCESS *****\n")
                    success_count += 1
                    break
                else:
                    print("\n***** OPTIMIZATION FAILED *****\n")
                    opt_weights = np.full(
                        (1, weights.shares_n), weights.shares_n)
                    break

            leaders_out_sample = np.array(
                out_sample_ret.T.iloc[weights.get_matched_rows(), :])
            out_sample_res = np.dot(opt_weights, leaders_out_sample)

            res.append(out_sample_res)
            shares_count.append(weights.shares_n)

            count += 1
            print("ATTEMPT {} OF {} COMPLETED. MOVING ON...".format(
                count, len(self.splits)))
            pd.DataFrame(out_sample_res).to_pickle(
                "./{}/replica_{}_{}.pkl".format(dir_global, count, self.date))
            pd.DataFrame(weights.get_matched_rows()).to_pickle(
                "./{}/replica_matched_{}_{}.pkl".format(dir_global, count, self.date))

        pd.DataFrame(shares_count).to_pickle(
            "./{}/shares_count_{}.pkl".format(dir_global, self.date))
        res_df = pd.DataFrame(np.concatenate(np.hstack(arr for arr in res)))
        res_df.to_pickle(
            "./{}/replica_{}.pkl".format(dir_global, self.date))
        return res_df, shares_count

    def runner_plot(self) -> plt.plot:
        """
        <DESCRIPTION>
        Plot out-sample results.
        """
        replica = pd.read_pickle(
            "./{}/replica_{}.pkl".format(dir_global, self.date)).cumsum()
        method = EmMethodFL(self.data_split.idx, DataSplit(
            self.mkt, self.date, self.idx_weight).stocks)
        original = method.stocks_ret.mean(
            axis=1)[freq_1:freq_1 + len(replica)].cumsum()

        shares_count = pd.read_pickle(
            "./{}/shares_count_{}.pkl".format(dir_global, self.date))
        shares_count = pd.DataFrame(np.concatenate(
            [item for item in shares_count.values for _ in range(freq_2)]))

        replica.index = original.index
        shares_count.index = original.index

        shares_count_diff = shares_count.diff()
        slope_change_idx = shares_count_diff.index[shares_count_diff[0] != 0]
        markevery = [shares_count.index.get_loc(i) for i in slope_change_idx]

        fig, ax1 = plt.subplots(figsize=(15, 5))
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
        ax2.legend(loc='best')

        # for i, index in enumerate(markevery):
        #     value = shares_count.iloc[index][0]
        #     ax2.annotate(str(value), (index, value), textcoords="offset points", xytext=(
        #         0, 10), ha='left', fontsize=8, color='black')
        plt.show()


if __name__ == "__main__":
    runner = MethodRunner(date='Y3')
    if os.path.exists('./{}/'.format(dir_global)):
        # res_df, shares_count = runner.runner()
        pass
    if os.path.exists('./{}/replica_{}.pkl'.format(dir_global, runner.date)):
        runner.runner_plot()
