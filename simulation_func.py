"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.pipeline as skpipe
import sklearn.decomposition as skd


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
    def func_bai_ng(data: np.ndarray, ic_method: int = 2, max_factor: int = 30):
        """
        <DESCRIPTION>
        Find the number of factors under Bai and Ng (2002).

        <NOTIFICATION>
        Code referred from: https://github.com/joe5saia/FredMD
        """
        T, N = data.shape
        NT = N * T
        NT1 = N + T

        if ic_method == 1:
            CT = [i * math.log(NT / NT1) * NT1 / NT for i in range(max_factor)]
        elif ic_method == 2:
            CT = [i * math.log(min(N, T)) * NT1 /
                  NT for i in range(max_factor)]
        elif ic_method == 3:
            CT = [i * math.log(min(N, T)) / min(N, T)
                  for i in range(max_factor)]
        else:
            raise ValueError("ic must be either 1, 2 or 3")

        pipe = skpipe.Pipeline(
            [('Factors', skd.TruncatedSVD(max_factor, algorithm='arpack'))])
        F = pipe.fit_transform(data)
        Lambda = pipe['Factors'].components_

        def V(X, F, Lambda):
            """
            <DESCRIPTION>
            Explained Variance of X by factors F with loadings Lambda.
            """
            T, N = X.shape
            NT = N*T
            return np.linalg.norm(X - F @ Lambda, 2)/NT

        Vhat = [V(data, F[:, 0:i], Lambda[0:i, :]) for i in range(max_factor)]
        IC = np.log(Vhat) + CT
        Nfactor = np.argmin(IC)
        return Nfactor

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
