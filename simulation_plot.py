"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
import matplotlib.pyplot as plt

from simulation_weights import *


class Plot(Weights):
    def __init__(self,
                 N: int = 100,
                 T: int = 1000,
                 k: int = 5,
                 EV: float = 0.99,
                 min_R2: float = 0.8):
        super().__init__(N, T, k, EV, min_R2)
        """
        <DESCRIPTION>
        Plots used for simulation.
        """
        if self.shares_n == 1:
            self.opt_weights = self.optimize()
        else:
            self.opt_weights, self.opt_res = self.optimize()
        self.idx_method = pd.DataFrame(self.shares)

    def plot_generator_price(self) -> plt.plot:
        """
        <DESCRIPTION>
        Plot generated price and its index.
        get_price() should be in progress before plot_price() is executed.
        """
        plt.figure(figsize=(15, 5))
        plt.plot(self.stock.T)
        plt.plot(self.idx, color='k', linewidth=5)

        plt.title("Generated prices")
        plt.xlabel("Time")
        plt.ylabel("Stocks")

        plt.show()

    def plot_method_price_eq_w(self):
        """
        <DESCRIPTION>
        Plot original index & replicated index by its price in equal weights.
        IN_SAMPLE.
        """
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
        """
        <DESCRIPTION>
        Plot original index & replicated index by its price in optimal weights.
        IN_SAMPLE.
        """
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
    plot = Plot()
    plot.plot_generator_price()
    plot.plot_method_price_eq_w()
    plot.plot_method_price_opt_w()
