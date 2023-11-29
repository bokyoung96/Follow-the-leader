"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from empirical_func import *
from empirical_loader import *
from empirical_plot_params import *


# 250, 5 TEST
start_date = '2011-01-01'
end_date = '2022-12-31'


def load_data():
    """
    <DESCRIPTION>
    Load VKOSPI data and others for observation.
    """
    idx = DataLoader(mkt='KOSPI200', date='Y15').as_empirical(
        idx_weight='EQ')[0].pct_change()
    replica = pd.read_pickle(
        './RUNNER_FL_Y15_p_val_0.1/RUNNER_FL_250_5_p_val_0.1/replica_Y15.pkl')
    replica_shares = pd.read_pickle(
        './RUNNER_FL_Y15_p_val_0.1/RUNNER_FL_250_5_p_val_0.1/shares_count_Y15.pkl')
    replica_real = pd.read_pickle(
        './RUNNER_FL_Y15_p_val_0.1_REAL/RUNNER_FL_250_5_p_val_0.1/replica_Y15.pkl')
    replica_real_shares = pd.read_pickle(
        './RUNNER_FL_Y15_p_val_0.1_REAL/RUNNER_FL_250_5_p_val_0.1/shares_count_Y15.pkl')
    vkospi = pd.read_pickle('./VKOSPI_DATA/VKOSPI.pkl')

    idx = idx[start_date:end_date]
    vkospi = vkospi[start_date:end_date]

    idx = (1 + Func().func_plot_init_price(idx, 100)).cumprod() - 1
    replica = (1 + Func().func_plot_init_price(replica, 100)).cumprod() - 1
    replica_real = (
        1 + Func().func_plot_init_price(replica_real, 100)).cumprod() - 1

    replica_shares = pd.DataFrame(np.concatenate(
        [item for item in replica_shares.values for _ in range(5)]))
    replica_real_shares = pd.DataFrame(np.concatenate(
        [item for item in replica_real_shares.values for _ in range(5)]))

    idx = idx[1:]
    replica = replica[1:]
    replica_real = replica_real[1:]
    return idx, replica, replica_shares, replica_real, replica_real_shares, vkospi


def pp_vkospi():
    """
    <DESCRIPTION>
    Preprocess VKOSPI data.
    """
    idx = load_data()[0]
    temp = pd.read_excel('./VKOSPI_DATA/VKOSPI.xlsx').set_index('Date')
    res = temp.loc[idx.index]
    res.to_pickle('./VKOSPI_DATA/VKOSPI.pkl')


def plot_data():
    """
    <DESCRIPTION>
    Plot VKOSPI data.
    """
    idx, replica, replica_shares, replica_real, replica_real_shares, vkospi = load_data()

    replica_shares = replica_shares[:replica.shape[0]]
    replica_real_shares = replica_real_shares[:replica_real.shape[0]]

    replica_shares.index = replica.index
    replica_real_shares.index = replica_shares.index

    replica_shares_diff = replica_shares.diff()
    slope_change_idx = replica_shares.index[replica_shares_diff[0] != 0]
    markevery = [replica_shares.index.get_loc(i) for i in slope_change_idx]

    replica_real_shares_diff = replica_real_shares.diff()
    slope_change_idx_real = replica_real_shares.index[replica_real_shares_diff[0] != 0]
    markevery_real = [replica_real_shares.index.get_loc(
        i) for i in slope_change_idx_real]

    vkospi_mean = round(vkospi.mean().values[0], 2)

    fig, ax1 = plt.subplots(figsize=(30, 10))
    plt.title('VKOSPI 지수와 리더 주식 개수의 관계: In 250, Out 5')

    ax1.plot(vkospi, label='VKOSPI 지수', color='black', linewidth=1.5)
    ax1.axhline(vkospi_mean, color='g', linestyle='--',
                label='VKSOPI 지수 평균: {}'.format(vkospi_mean), linewidth=1)
    ax1.set_xlabel('년도')
    ax1.set_ylabel('지수 값')
    ax1.legend(loc='upper right')

    ax2 = ax1.twinx()
    ax2.plot(replica_real_shares, label='리더 주식 개수, FL',
             color='r', linewidth=1, linestyle='--')
    ax2.plot(replica_shares, label='리더 주식 개수, FL-Adjusted',
             color='b', linewidth=1, linestyle='--')
    ax2.set_ylabel('종목 개수')
    ax2.legend(loc='upper center')

    # plt.show()
    plt.savefig(
        './RUNNER_GRAPHS_ETC/plot_vkospi.jpg',
        format='jpeg',
        bbox_inches='tight')


def vkospi_vol(start_slice, end_slice):
    """
    <DESCRIPTION>
    Calculate VKOSPI mean and volatility in in-sample for upcoming out-sample results.
    """
    vkospi = pd.read_pickle('./VKOSPI_DATA/VKOSPI.pkl')[start_date:end_date]

    temp = vkospi[start_slice:end_slice]
    window_size = 250
    start_idx = 0
    end_idx = start_idx + window_size

    rolling_std_list = []
    rolling_mean_list = []
    while end_idx < len(temp):
        window_data = temp.iloc[start_idx:end_idx]

        rolling_std = window_data.std()
        rolling_std_list.append(rolling_std)

        rolling_mean = window_data.mean()
        rolling_mean_list.append(rolling_mean)

        start_idx += 5
        end_idx = start_idx + window_size

    res_mean = np.round(pd.DataFrame(rolling_mean_list), 4)
    res_std = np.round(pd.DataFrame(rolling_std_list), 4)
    res = pd.concat([res_mean, res_std], axis=1)
    res.columns = ['MEAN', 'STD.DEV']
    return res


if __name__ == "__main__":
    plot_data()
    # res = vkospi_vol('2017-07-01', '2019-01-03')
