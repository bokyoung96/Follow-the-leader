"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from simulation_weights_CM import *
from simulation_weights_FL import *


class PlotAbstr(ABC):
    """
    <DESCRIPTION>
    Abstract method of Plot for CM-2006 and FL methods.
    """
    @abstractmethod
    def plot_generator_price(self):
        """
        <DESCRIPTION>
        Plot generated price and its index.
        get_price() should be in progress before plot_price() is executed.
        """
        pass

    @abstractmethod
    def plot_method_price_eq_w(self):
        """
        <DESCRIPTION>
        Plot original index & replicated index by its price in equal weights.
        IN_SAMPLE.
        """
        pass

    @abstractmethod
    def plot_method_price_opt_w(self):
        """
        <DESCRIPTION>
        Plot original index & replicated index by its price in optimal weights.
        IN_SAMPLE.
        """
        pass


class PlotCM(PlotAbstr, WeightsCM):
    def __init__(self,
                 N: int = 100,
                 T: int = 1000,
                 k: int = 5,
                 EV: float = 0.99,
                 min_R2: float = 0.8):
        """
        <DESCRIPTION>
        Plots used for simulation under CM-2006 method.

        <PARAMETER>
        Same as WeightsCM class.

        <CONSTRUCTOR>
        self.opt_weights, self.opt_res: Optimal results from self.optimize() in Weights class.
        self.idx_method: Selected shares.

        <NOTIFICATION>
        Function explanation is written in abstract method above.
        """
        super().__init__(N, T, k, EV, min_R2)
        if self.shares_n == 1:
            self.opt_weights = self.optimize()
        else:
            self.opt_weights, self.opt_res = self.optimize()
        self.idx_method = pd.DataFrame(self.shares)

    def plot_generator_price(self) -> plt.plot:
        plt.figure(figsize=(15, 5))
        plt.plot(self.stock.T)
        plt.plot(self.idx, color='k', linewidth=5)

        plt.title("Generated prices")
        plt.xlabel("Time")
        plt.ylabel("Stocks")

        plt.show()

    def plot_method_price_eq_w(self):
        idx = self.idx_in_sample
        idx_method = self.idx_method.mean()

        plt.figure(figsize=(15, 5))
        plt.plot(idx, label="Original index")
        plt.plot(idx_method, label="Replicated index")

        plt.title(
            "Original index versus Replicated index in equal weights (IN-SAMPLE)")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend(loc='best')

        plt.show()

    def plot_method_price_opt_w(self):
        idx = self.idx_in_sample
        if self.shares_n == 1:
            idx_method = self.shares
        else:
            idx_method = np.dot(self.opt_weights, self.shares).flatten()

        plt.figure(figsize=(15, 5))
        plt.plot(idx, label='Original index')
        plt.plot(idx_method, label='Replicated index')

        plt.title(
            'Original index versus Replicated index in optimial weights (IN-SAMPLE)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc='best')

        plt.show()


class PlotFL(PlotAbstr, WeightsFL):
    def __init__(self,
                 N: int = 100,
                 T: int = 1000,
                 k: int = 5,
                 F_max: int = 30,
                 p_val: float = 0.05):
        """
        <DESCRIPTION>
        Plots used for simulation under FL method.

        <PARAMETER>
        Same as WeightsFL class.

        <CONSTRUCTOR>
        self.opt_weights, self.opt_res: Optimal results from self.optimize() in Weights class.
        self.idx_method: Selected shares.

        <NOTIFICATION>
        Function explanation is written in abstract method above.
        """
        super().__init__(N, T, k, F_max, p_val)
        if self.shares_n == 1:
            self.opt_weights = self.optimize()
        else:
            self.opt_weights, self.opt_res = self.optimize()
        self.idx_method = pd.DataFrame(self.shares)

    def plot_generator_price(self) -> plt.plot:
        plt.figure(figsize=(15, 5))
        plt.plot(self.stock.T)
        plt.plot(self.idx, color='k', linewidth=5)

        plt.title("Generated prices")
        plt.xlabel("Time")
        plt.ylabel("Stocks")

        plt.show()

    def plot_method_price_eq_w(self):
        idx = self.idx_in_sample
        idx_method = self.idx_method.mean()

        plt.figure(figsize=(15, 5))
        plt.plot(idx, label="Original index")
        plt.plot(idx_method, label="Replicated index")

        plt.title(
            "Original index versus Replicated index in equal weights (IN-SAMPLE)")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend(loc='best')

        plt.show()

    def plot_method_price_opt_w(self):
        idx = self.idx_in_sample
        if self.shares_n == 1:
            idx_method = self.shares
        else:
            idx_method = np.dot(self.opt_weights, self.shares).flatten()

        plt.figure(figsize=(15, 5))
        plt.plot(idx, label='Original index')
        plt.plot(idx_method, label='Replicated index')

        plt.title(
            'Original index versus Replicated index in optimial weights (IN-SAMPLE)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc='best')

        plt.show()


if __name__ == "__main__":
    plotfl = PlotFL()
    plotfl.plot_generator_price()
    plotfl.plot_method_price_eq_w()
    plotfl.plot_method_price_opt_w()
