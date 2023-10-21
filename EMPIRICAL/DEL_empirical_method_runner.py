"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
from pathlib import Path

from DEL_empirical_weights_FL import *


# LOCATE DIRECTORY
def locate_dir(dir_name):
    path = Path(dir_name)
    path.mkdir(parents=True, exist_ok=True)


locate_dir('./RUNNER_p_val/')


class DataSplit:
    def __init__(self, mkt: str = 'KOSPI200',
                 date: str = 'Y3',
                 idx_weight: str = 'EQ'):
        self.data_loader = DataLoader(mkt, date)
        self.idx, self.stocks = self.data_loader.as_empirical(
            idx_weight=idx_weight)
        self.months = 20
        self.years = 250

    def data_split(self):
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
                 p_val: float = 0.01,
                 F_max: int = 30,
                 EV: float = 0.9,
                 mkt: str = 'KOSPI200',
                 date: str = 'Y10',
                 idx_weight: str = 'EQ'
                 ):
        self.p_val = p_val
        self.F_max = F_max
        self.EV = EV
        self.mkt = mkt
        self.date = date
        self.months = 20
        self.years = 250

        self.data_split = DataSplit(mkt, date, idx_weight)
        self.splits = self.data_split.data_split()

    @time_spent_decorator
    def runner(self):
        res = []
        count = 0
        shares_count = []
        success_count = 0
        count = 0
        sample_division = -20
        for stocks in self.splits:
            in_sample = stocks.iloc[:sample_division]
            out_sample_ret = stocks.pct_change().iloc[sample_division:]

            if stocks.shape[0] < self.years + self.months - 1:
                print("ITERATION LIMIT REACHED. FINISHING...")
                break

            while True:
                weights = EmWeightsFL(self.data_split.idx,
                                      in_sample,
                                      self.F_max,
                                      self.EV,
                                      self.p_val)
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
                "./RUNNER_p_val_scaled/replica_{}_{}.pkl".format(count, self.date))

        pd.DataFrame(shares_count).to_pickle(
            "./RUNNER_p_val_scaled/shares_count_{}.pkl".format(self.date))
        res_df = pd.DataFrame(np.concatenate(np.hstack(arr for arr in res)))
        res_df.to_pickle(
            "./RUNNER_p_val_scaled/replica_{}.pkl".format(self.date))
        return res, shares_count


if __name__ == "__main__":
    runner_3Y = MethodRunner(date='Y3')
    res, shares_count = runner_3Y.runner()
