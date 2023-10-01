"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
import numpy as np
import pandas as pd

from simulation_loader import DataLoader


class Analysis:
    def __init__(self, iters: int = 1000, method_type: str = 'FL'):
        """
        <DESCRIPTION>
        Analyze the data from CM-2006 method.

        <PARAMETER>
        iters: Number of simulation iteration.
        method_type: Type of method used.

        <CONSTRUCTOR>
        data_loader: Instance of DataLoader class to call datas.
        self.in_sample, self.out_sample: Data after simulation, classified by in and out samples.
        self.in_sample_msres, self.out_sample_msres: Final data for table 1 in the article.
        """
        self.iters = iters
        self.method_type = method_type
        data_loader = DataLoader()

        if self.method_type == 'CM':
            self.in_sample = data_loader.get_data_cm_2006(sample_type='in')
            self.out_sample = data_loader.get_data_cm_2006(sample_type='out')
        elif self.method_type == 'FL':
            self.in_sample = data_loader.get_data_fl(sample_type='in')
            self.out_sample = data_loader.get_data_fl(sample_type='out')
        else:
            raise AssertionError("ERROR: Select between [CM, FL].")

        self.chg_cols()

        self.in_sample_msres = None
        self.out_sample_msres = None
        self.table_1_results()

    @property
    def in_sample_cols(self) -> list:
        """
        <DESCRIPTION>
        In-sample columns.
        """
        return ['Nfactors', 'Nleaders', 'MEAN', 'STD.DEV.', 'MAD', 'SUPMOD', 'MSE']

    @property
    def out_sample_cols(self) -> list:
        """
        <DESCRIPTION>
        Out-sample columns.
        """
        return ['MEAN', 'STD.DEV.', 'MAD', 'SUPMOD', 'MSE', 'Correl.']

    def chg_cols(self):
        """
        <DESCRIPTION>
        Change columns of the data under in_sample_cols and out_sample_cols.
        """
        for df in self.in_sample:
            df.columns = self.in_sample_cols
        for df in self.out_sample:
            df.columns = self.out_sample_cols

    def table_1_msres(self, df: pd.DataFrame):
        """
        <DESCRIPTION>
        Calculate the mean and standard error of the performance measure from the result.
        Change its columns to value and standard error.
        """
        res = pd.concat([df.mean(), df.std() / np.sqrt(self.iters)], axis=1)
        res.columns = ['Value', 'Standard_Error']
        return res

    def table_1_results(self):
        """
        <DESCRIPTION>
        Print table 1.
        Table 1 can be checked as self.in_sample_msres and self.out_sample_msres.
        """
        self.in_sample_msres = []
        for df in self.in_sample:
            self.in_sample_msres.append(self.table_1_msres(df))

        self.out_sample_msres = []
        for df in self.out_sample:
            self.out_sample_msres.append(self.table_1_msres(df))


if __name__ == "__main__":
    analysis = Analysis()
