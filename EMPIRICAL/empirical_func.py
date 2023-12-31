"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA, TruncatedSVD


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
            CT = [i * math.log(NT / NT1) * NT1 /
                  NT for i in range(max_factor+1)]
        elif ic_method == 2:
            CT = [i * math.log(min(N, T)) * NT1 /
                  NT for i in range(max_factor+1)]
        elif ic_method == 3:
            CT = [i * math.log(min(N, T)) / min(N, T)
                  for i in range(max_factor+1)]
        else:
            raise ValueError("ic must be either 1, 2 or 3")

        truncated_svd = TruncatedSVD(
            n_components=max_factor, algorithm='arpack')
        F = truncated_svd.fit_transform(data)
        Lambda = truncated_svd.components_

        def V(X, F, Lambda):
            """
            <DESCRIPTION>
            Explained Variance of X by factors F with loadings Lambda.
            """
            residual_matrix = X - F @ Lambda
            return np.linalg.norm(residual_matrix, 'fro') ** 2 / NT

        Vhat = [V(data, F[:, 0:i], Lambda[0:i, :])
                for i in range(0, max_factor+1)]
        IC = np.log(Vhat) + np.array(CT)
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
    def func_diagonal(resid_matrix: pd.DataFrame, tolerance: float = 1e-4) -> bool:
        """
        <DESCRIPTION>
        Test whether the covariance matrix of residuals is diagonal.
        """
        resid_matrix_np = resid_matrix.to_numpy()
        diagonal_elements = np.diag(resid_matrix_np)

        expanded_diagonal_matrix = np.zeros_like(resid_matrix_np)
        np.fill_diagonal(expanded_diagonal_matrix, diagonal_elements)

        non_diag_elements = np.abs(resid_matrix_np - expanded_diagonal_matrix)

        is_diag = np.all(non_diag_elements < tolerance)
        return is_diag

    @staticmethod
    def func_plot_init_price(df: pd.DataFrame, init_price: int) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Add initial price when plotting.
        """
        return pd.concat([pd.Series(init_price), df])
