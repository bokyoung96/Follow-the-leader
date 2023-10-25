"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import numpy as np
import pandas as pd
from enum import Enum, unique, auto

from time_spent_decorator import time_spent_decorator


DIR_DATA = "_DATA_"


@unique
class DirMkt(Enum):
    KOSPI = auto()
    KOSPI100 = auto()
    KOSPI200 = auto()
    KOSDAQ = auto()
    KOSDAQ150 = auto()


@unique
class DirDate(Enum):
    Y1 = auto()
    Y3 = auto()
    Y5 = auto()
    Y10 = auto()
    Y20 = auto()
    WHOLE = auto()


class DataLoader:
    def __init__(self, mkt: str = 'KOSPI200', date: str = 'Y1'):
        """
        <DESCRIPTION>
        Load price data under DataGuide xlsx files.
        """
        print("DATA LOADING...\n")
        print("MKT: {}\n".format(mkt))
        print("FREQ: {}\n".format(date))

        mkt_keys = DirMkt.__members__.keys()
        date_keys = DirDate.__members__.keys()

        assert mkt in mkt_keys, "Invalid market value. Insert {}.".format(
            mkt_keys)
        assert date in date_keys, "Invalid date value. Insert {}.".format(
            date_keys)

        mkt_value = getattr(DirMkt, mkt).name
        date_value = getattr(DirDate, date).name

        self.idx_name = mkt
        self.file_name = f"./{mkt_value}{DIR_DATA}{date_value}.xlsx"
        self.sheet_name = ["IDX", "CURRENT", "ALL"]

    def get_raw_data(self, sheet_name: str = 'IDX') -> pd.DataFrame:
        """
        <DESCRIPTION>
        Load price data under sheet_name. (IDX, CURRENT, ALL)
        """
        return pd.read_excel(self.file_name, sheet_name=sheet_name, header=8).iloc[5:, :]

    @property
    def get_data(self) -> list:
        """
        <DESCRIPTION>
        Preprocess price datas.
        """
        datas = []
        for item in self.sheet_name:
            data = self.get_raw_data(sheet_name=item)

            # NOTE: Drop process will be in runner due to avoid bias.
            # data.dropna(how='any', axis=1, inplace=True)
            data = data.rename(columns={'Symbol': 'Date'})
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)

            datas.append(data)
        return datas

    @time_spent_decorator
    def as_empirical(self, idx_weight: str = "N") -> pd.DataFrame:
        """
        <DESCRIPTION>
        Load price data adjusting weighting scheme for index.

        <NOTIFICATION>
        Check NOTE to determine constituent data.
        1: CURRENT
        2: ALL
        """
        datas = self.get_data

        if idx_weight == "EQ":
            self.idx_name = f"{self.idx_name}_EQ"
        else:
            pass

        idx = datas[0][self.idx_name].astype(np.float64)
        # NOTE: UPLOAD <2: ALL> FOR BIAS AVOIDING
        stocks = datas[2]
        # NOTE: Drop process will be in runner due to avoid bias.
        # for col in stocks.columns:
        #     if stocks[col].nunique() == 1:
        #         stocks = stocks.drop(col, axis=1)

        print("DATA LOADED. IDX WEIGHTS ARE: {}\n".format(idx_weight))
        return idx, stocks

    @time_spent_decorator
    def fast_as_empirical(self, idx_weight: str = "N") -> pd.DataFrame:
        """
        <DESCRIPTION>
        Load price data adjusting weighting scheme for index.

        <NOTIFICATION>
        Fast version. Bias not considered.
        """
        datas = self.get_data

        if idx_weight == "EQ":
            self.idx_name = f"{self.idx_name}_EQ"
        else:
            pass

        idx = datas[0][self.idx_name].astype(np.float64)
        stocks = datas[1]
        stocks = stocks.dropna(how='any', axis=1)
        for col in stocks.columns:
            if stocks[col].nunique() == 1:
                stocks = stocks.drop(col, axis=1)

        print("DATA LOADED. IDX WEIGHTS ARE: {}\n".format(idx_weight))
        return idx, stocks


if __name__ == "__main__":
    data_loader = DataLoader(mkt='KOSPI200', date='Y3')
    idx, stocks = data_loader.as_empirical(idx_weight='EQ')
