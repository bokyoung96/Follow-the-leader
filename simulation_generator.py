"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
import numpy as np
import pandas as pd
import timeout_decorator
from tqdm import tqdm

# Seed fixing
# np.random.seed(42)


class Generator:
    def __init__(self, N: int = 100, T: int = 1000, k: int = 5):
        """
        <DESCRIPTION>
        Generate required values to process Monte-Carlo simulation.

        <PARAMETER>
        N: Number of stocks.
        T: Final timestamp.
        k: Number of factors.

        <CONSTRUCTOR>
        F_random_walk, F_stationary, FL, stock: Will be constructed in get_price function.
        idx: Will be constructed in get_idx function.
        get_price, get_idx: Initial function in progress for constructors.
        """
        self.N = N
        self.T = T
        self.k = k

        self.F_random_walk = None
        self.F_stationary = None
        self.FL = None
        self.stock = None
        self.idx = None

        self.get_price
        self.get_idx
        print("Price and Index is generated successfully.")
        print("Do not recall get_price and get_idx function.")

        print("\n ***** VALUES ***** \n")
        print(self)

    def __str__(self) -> str:
        """
        <DESCRIPTION>
        Magic method to check self objects in __init__.
        """
        attr = [f"{key} : {value}" for key, value in self.__dict__.items()]
        return "\n".join(attr)

    @property
    def generator_error(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Generate error terms following normal distribution.
        """
        return np.random.normal(0, 1, size=(self.k, self.T))

    @property
    def generator_factor_loading(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Generate factor loadings following normal distribution.
        """
        return np.random.normal(0, 1, size=(1, self.k))

    def generator_factor(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Generate factors.
        Factors classified into 2 groups: Random Walk, Stationary.
        """
        F_random_walk = np.cumsum(self.generator_error, axis=1)
        F_stationary = self.generator_error
        return F_random_walk, F_stationary

    @timeout_decorator.timeout(15)
    def generator_positive_price(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Generate positive stock price.
        """
        F_random_walk, F_stationary = self.generator_factor()

        prices = []
        FL = []
        with tqdm(total=self.N) as pbar:
            n = 0
            while n < self.N:
                FL_random_walk = self.generator_factor_loading
                FL_stationary = self.generator_factor_loading

                random_walk = np.matmul(FL_random_walk, F_random_walk)
                stationary = np.matmul(FL_stationary, F_stationary)
                err = np.random.normal(0, 1, size=(1, self.T))

                price = random_walk + stationary + err
                if (price >= 0).all():
                    prices.append(price)
                    FL.append(
                        np.hstack([FL_random_walk, FL_stationary]).flatten())
                else:
                    continue
                pbar.update(1)
                n += 1
        FL = np.array(FL)
        return prices, F_random_walk, F_stationary, FL

    @property
    def get_price(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Get price dataframe.
        Price and factors will be added as constructor due to the randomizing applied in the generator.

        <CAUTION>
        Do not use get_price function under other functions.
        Do not call get_price function more than once under other environments.
        Constructor and get_price variables will change and show difference.

        <NOTIFICATION>
        If generator_positive_price() does not reach for the output for 15 seconds, it will restart.
        """
        while True:
            try:
                prices, F_random_walk, F_stationary, FL = self.generator_positive_price()
                break
            except:
                print("ERROR: TIME-OUT. GENERATOR RESTARTING...")

        dfs = [pd.DataFrame(prices[n]) for n in range(self.N)]
        res = pd.concat(dfs, ignore_index=True)

        self.stock = res
        self.F_random_walk = F_random_walk
        self.F_stationary = F_stationary
        self.FL = FL

    @property
    def get_idx(self) -> pd.Series:
        """
        <DESCRIPTION>
        Get idx series.
        get_price() should be in progress before get_idx() is executed.
        """
        idx = self.stock.mean()

        self.idx = idx


if __name__ == "__main__":
    gen = Generator(N=100, T=1000, k=5)
