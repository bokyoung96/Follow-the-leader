import os
import pickle
import itertools
import numpy as np
import pandas as pd

from enum import Enum, unique

DIR_DATA = "DATA_%s"
MSRES = "_MSRES_%s"


@unique
class DirFolder(Enum):
    CM_2006 = "CM_2006/"
    FL = "FL/"

    def as_dir(self):
        return DIR_DATA % self.value


@unique
class DirFileCM(Enum):
    EV90 = "EV90%s"
    EV99 = "EV99%s"
    # EV999 = "EV999%s"

    def as_in(self):
        return self.value % MSRES % "IN"

    def as_out(self):
        return self.value % MSRES % "OUT"


class DataPool:
    data_cm_2006 = DirFolder.CM_2006.as_dir()
    data_fl = DirFolder.FL.as_dir()

    in_sample_cm = [DirFileCM(member).as_in() for member in DirFileCM]
    out_sample_cm = [DirFileCM(member).as_out() for member in DirFileCM]

    def as_cm_2006_in(self):
        return [self.data_cm_2006 + item for item in self.in_sample_cm]

    def as_cm_2006_out(self):
        return [self.data_cm_2006 + item for item in self.out_sample_cm]


class DataLoader(DataPool):
    def __init__(self):
        self.CM_2006_IN = self.as_cm_2006_in()
        self.CM_2006_OUT = self.as_cm_2006_out()
        message = """ 
        NOTIFICATION: (get_data_cm_2006) CM_2006 DATA ORDERS IN EV CUTOFF 90, 99.\n\n
        NOTIFICATION: (get_data) CM_2006 DATA ORDERS IN IN, OUT, EV CUTOFF 90, 99.
        """
        print(message)

    def get_data_cm_2006(self, sample_type: str = 'in') -> pd.DataFrame:
        if sample_type == 'in':
            return [pd.read_pickle("./{}.pkl".format(data)) for data in self.CM_2006_IN]
        elif sample_type == 'out':
            return [pd.read_pickle("./{}.pkl".format(data)) for data in self.CM_2006_OUT]

    @property
    def get_data(self) -> pd.DataFrame:
        datas = list(itertools.chain.from_iterable(self.__dict__.values()))
        return [pd.read_pickle("./{}.pkl".format(data)) for data in datas]


if __name__ == "__main__":
    data_loader = DataLoader()
    cm_2006 = data_loader.get_data_cm_2006(sample_type='in')
    data = data_loader.get_data