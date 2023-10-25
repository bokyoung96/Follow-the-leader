"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import os
import numpy as np
import pandas as pd
from enum import Enum, unique, auto

from time_spent_decorator import time_spent_decorator


DIR_CONST = "_CONST_"


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


class ConstLoader:
    def __init__(self, mkt: str = 'KOSPI200', date: str = 'Y1'):
        """
        <DESCRIPTION>
        Load constituent data under DataGuide xlsx files.
        데이터 센터 - 항목별 데이터 - 기업 이벤트 - 구성종목 내역조회
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

        self.base_file_name = f"{mkt_value}_CONST"
        self.file_name = f"./{self.base_file_name}/{mkt_value}{DIR_CONST}{date_value}.xlsx"
        self.sheet_name = ["Constituents"]

    def get_raw_data(self, sheet_name: str = 'Constituents') -> pd.DataFrame:
        """
        <DESCRIPTION>
        Load constituent data under sheet_name. (Constituents)
        """
        data = pd.read_excel(self.file_name, sheet_name=sheet_name, header=6)
        data.columns = ['Date', 'Code', 'Name']
        return data

    @time_spent_decorator
    def get_pivot_data(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Preprocess constituent data.
        """
        data = self.get_raw_data(sheet_name='Constituents')

        data = data.drop(['Name'], axis=1)
        temp = data.pivot(index='Date', columns='Code', values='Code')
        res = temp.notna().astype(int)
        return res

    @property
    def save_pivot_data(self):
        """
        <DESCRIPTION>
        Save constituent data to pkl file for time-saving.
        """
        file_name = os.path.splitext(self.file_name)[0]
        return self.get_pivot_data().to_pickle(f"{file_name}_PIVOT.pkl")


if __name__ == "__main__":
    const_loader = ConstLoader(mkt='KOSPI200', date='Y5')
    # data = const_loader.get_pivot_data()

    splitter = os.path.splitext(const_loader.file_name)[0]
    file_name = "{}_PIVOT.pkl".format(splitter)
    if os.path.exists("./{}".format(file_name)):
        pass
    else:
        const_loader.save_pivot_data
