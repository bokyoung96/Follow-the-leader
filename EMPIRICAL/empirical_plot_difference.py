"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from empirical_func import *
from empirical_loader import *


start_date = '2011-01-01'
end_date = '2022-12-31'


def load_data(date: str = 'Y15',
              freq_1: int = 250,
              freq_2: int = 20) -> pd.DataFrame:
    """
    <DESCRIPTION>
    Load each data to plot.
    """
    CM_90 = pd.read_pickle(
        f"RUNNER_CM_{date}_ev_0.9/RUNNER_CM_{freq_1}_{freq_2}_ev_0.9/replica_{date}.pkl")
    CM_99 = pd.read_pickle(
        f"RUNNER_CM_{date}_ev_0.99/RUNNER_CM_{freq_1}_{freq_2}_ev_0.99/replica_{date}.pkl")
    FL = pd.read_pickle(
        f"RUNNER_FL_Y15_p_val_0.1/RUNNER_FL_{freq_1}_{freq_2}_p_val_0.1/replica_{date}.pkl")
    NV = pd.read_pickle(
        f"RUNNER_NV_Y15_num_100/RUNNER_NV_{freq_1}_{freq_2}_num_100/replica_{date}.pkl")
    return CM_90, CM_99, FL, NV


def load_plot_data(date: str = 'Y15',
                   freq_1: int = 250,
                   freq_2: int = 20) -> pd.DataFrame:
    """
    <DESCRIPTION>
    Load each data to plot.
    """
    CM_90, CM_99, FL, NV = load_data(date=date,
                                     freq_1=freq_1,
                                     freq_2=freq_2)
    loader = DataLoader(mkt='KOSPI200',
                        date=date)
    idx = loader.as_empirical(idx_weight='EQ')[0]
    idx_ret = idx.pct_change().loc[start_date:end_date]

    CM_90 = (1 + Func().func_plot_init_price(CM_90, 100)).cumprod() - 1
    CM_99 = (1 + Func().func_plot_init_price(CM_99, 100)).cumprod() - 1
    FL = (1 + Func().func_plot_init_price(FL, 100)).cumprod() - 1
    NV = (1 + Func().func_plot_init_price(NV, 100)).cumprod() - 1
    idx_ret = (1 + Func().func_plot_init_price(idx_ret, 100)).cumprod() - 1

    idx_ret.rename(
        index={0: idx_ret.index[1] - pd.DateOffset(days=1)}, inplace=True)

    CM_90.index = idx_ret.index
    CM_99.index = idx_ret.index
    FL.index = idx_ret.index
    NV.index = idx_ret.index
    return CM_90, CM_99, FL, NV, idx_ret


def plot_data(date: str = 'Y15',
              freq_1: int = 250,
              freq_2: int = 20) -> plt.plot:
    """
    <DESCRIPTION>
    Plot each datas.
    """
    CM_90, CM_99, FL, NV, idx_ret = load_plot_data(date, freq_1, freq_2)

    plt.figure(figsize=(15, 5))
    plt.title('Plots of each methodologies: {}, {}'.format(freq_1, freq_2))
    plt.plot(CM_90, label='CM-2006 (EV 90%)')
    plt.plot(CM_99, label='CM-2006 (EV 99%)')
    plt.plot(FL, label='Follow the leader')
    plt.plot(NV, label='Naive correlation')
    plt.plot(idx_ret, label='KOSPI200 EQ Index', color='black', linewidth=2)

    plt.xlabel('Date')
    plt.ylabel('Cumulative return')

    plt.legend(loc='best')
    plt.show()


def plot_difference(date: str = 'Y15',
                    freq_1: int = 250,
                    freq_2: int = 20) -> plt.plot:
    """
    <DESCRIPTION>
    Plot the difference of each datas.
    """
    CM_90, CM_99, FL, NV, idx_ret = load_plot_data(date, freq_1, freq_2)

    x = FL.index

    CM_90_diff = np.abs(idx_ret.values - CM_90.values.flatten())
    CM_99_diff = np.abs(idx_ret.values - CM_99.values.flatten())
    FL_diff = np.abs(idx_ret.values - FL.values.flatten())
    NV_diff = np.abs(idx_ret.values - NV.values.flatten())

    diff_sum = CM_90_diff + CM_99_diff + FL_diff + NV_diff
    diff_sum[0] = 1

    CM_90_diff = CM_90_diff / diff_sum
    CM_99_diff = CM_99_diff / diff_sum
    FL_diff = FL_diff / diff_sum
    NV_diff = NV_diff / diff_sum

    fig, ax1 = plt.subplots(figsize=(15, 5))

    ax1.set_title(
        'Difference of each methodologies: {}, {}'.format(freq_1, freq_2))
    ax1.plot(x, idx_ret, label='KOSPI200 EQ Index',
             color='black', linewidth=2.5, alpha=1, zorder=10)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative return', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.fill_between(x, 0, CM_90_diff, label='CM-2006 (EV 90%)', alpha=0.5)
    ax2.fill_between(x, CM_90_diff, CM_90_diff + CM_99_diff,
                     label='CM-2006 (EV 99%)', alpha=0.5)
    ax2.fill_between(x, CM_90_diff + CM_99_diff, CM_90_diff +
                     CM_99_diff + FL_diff, label='Follow the leader', alpha=0.5)
    ax2.fill_between(x, CM_90_diff + CM_99_diff + FL_diff, CM_90_diff +
                     CM_99_diff + FL_diff + NV_diff, label='Naive correlation', alpha=0.5)
    ax2.set_ylabel('Difference Value')

    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # plot_data()
    plot_difference()
