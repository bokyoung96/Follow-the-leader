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
import sklearn.preprocessing as skp
from sklearn.decomposition import PCA


class Func:
    def __init__(self) -> None:
        """
        <DESCRIPTION>
        Functions used for simulation.
        """
        pass

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

        # scaler = skp.StandardScaler()
        # data = scaler.fit_transform(data)

        pipe = skpipe.Pipeline(
            [('Factors', skd.TruncatedSVD(max_factor, algorithm='arpack'))])
        F = pipe.fit_transform(data)
        Lambda = pipe['Factors'].components_

        # pca = PCA(n_components=max_factor)
        # F = pca.fit_transform(data)
        # Lambda = pca.components_

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
        return Nfactor, IC[Nfactor]

    @staticmethod
    def func_pca(n_components: int, df: pd.DataFrame) -> np.ndarray:
        """
        <DESCRIPTION>
        Do PCA analysis for factor extraction.
        """
        pca = PCA(n_components=n_components)
        PC = pca.fit_transform(df)
        return pca, PC

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

    @staticmethod
    def func_diagonal(resid_matrix: np.ndarray, tolerance: float = 1e-4) -> bool:
        """
        <DESCRIPTION>
        Test whether the covariance matrix of residuals is diagonal.
        """
        non_diag_elements = np.abs(
            resid_matrix - np.diag(np.diagonal(resid_matrix)))
        is_diag = np.all(non_diag_elements < tolerance)
        return is_diag
