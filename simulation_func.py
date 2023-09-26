"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


class Func:
    def __init__(self) -> None:
        """
        <DESCRIPTION>
        Functions used for simulation.
        """
        pass

    @staticmethod
    def func_split_stocks(df) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Divide dataframe / series into two parts under columns - in-sample and out-sample.
        """
        if isinstance(df, pd.DataFrame):
            cols = df.shape[1]
            half_cols = cols // 2

            in_sample = df.iloc[:, :half_cols]
            out_sample = df.iloc[:, half_cols:]
        elif isinstance(df, pd.Series):
            cols = len(df)
            half_cols = cols // 2

            in_sample = df[:half_cols]
            out_sample = df[half_cols:]
        return in_sample, out_sample

    @staticmethod
    def func_split_factors(temp: np.ndarray) -> np.ndarray:
        """
        <DESCRIPTION>
        Split each factors into two ndarray for in-sample and out-sample divison.
        """
        return np.array_split(temp, 2, axis=1)[0]

    @staticmethod
    def func_rank(temp: list) -> list:
        """
        <DESCRIPTION>
        Rank given temp by its values in ascending order.
        """
        res = [sorted(temp, reverse=True).index(x) for x in temp]
        return res

    @staticmethod
    def func_regression(x: pd.Series, y: np.ndarray) -> sm.OLS:
        """
        <DESCRIPTION>
        Do OLS for given x and y.
        """
        x = sm.add_constant(x)
        return sm.OLS(y, x).fit()
