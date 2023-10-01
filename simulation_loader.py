"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
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
    EV999 = "EV999%s"

    def as_in(self):
        return self.value % MSRES % "IN"

    def as_out(self):
        return self.value % MSRES % "OUT"


@unique
class DirFileFL(Enum):
    FL005 = "FL0.05%s"
    # FL001 = "FL0.01%s"

    def as_in(self):
        return self.value % MSRES % "IN"

    def as_out(self):
        return self.value % MSRES % "OUT"


class DataPool:
    """
    <DESCRIPTION>
    Call data directory.
    """
    data_cm_2006 = DirFolder.CM_2006.as_dir()
    data_fl = DirFolder.FL.as_dir()

    in_sample_cm = [DirFileCM(member).as_in() for member in DirFileCM]
    out_sample_cm = [DirFileCM(member).as_out() for member in DirFileCM]

    in_sample_fl = [DirFileFL(member).as_in() for member in DirFileFL]
    out_sample_fl = [DirFileFL(member).as_out() for member in DirFileFL]

    def as_cm_2006_in(self):
        return [self.data_cm_2006 + item for item in self.in_sample_cm]

    def as_cm_2006_out(self):
        return [self.data_cm_2006 + item for item in self.out_sample_cm]

    def as_fl_in(self):
        return [self.data_fl + item for item in self.in_sample_fl]

    def as_fl_out(self):
        return [self.data_fl + item for item in self.out_sample_fl]


class DataLoader(DataPool):
    def __init__(self):
        """
        <DESCRIPTION>
        Call data from its directory.

        <CONSTRUCTOR>
        CM_2006_IN, CM_2006_OUT: Call data directories.
        """
        self.CM_2006_IN = self.as_cm_2006_in()
        self.CM_2006_OUT = self.as_cm_2006_out()
        self.FL_IN = self.as_fl_in()
        self.FL_OUT = self.as_fl_out()
        message = """ 
        NOTIFICATION: (get_data_cm_2006) CM_2006 DATA ORDERS IN EV CUTOFF 90, 99, 99.9.\n\n
        NOTIFICATION: (get_data) CM_2006 DATA ORDERS IN IN, OUT, EV CUTOFF 90, 99, 99.9.
        """
        print(message)

    def get_data_cm_2006(self, sample_type: str = 'in') -> pd.DataFrame:
        """
        <DESCRIPTION>
        Get datas generated from CM-2006 method.
        """
        if sample_type == 'in':
            return [pd.read_pickle("./{}.pkl".format(data)) for data in self.CM_2006_IN]
        elif sample_type == 'out':
            return [pd.read_pickle("./{}.pkl".format(data)) for data in self.CM_2006_OUT]

    def get_data_fl(self, sample_type: str = 'in') -> pd.DataFrame:
        """
        <DESCRIPTION>
        Get datas generated from FL method.
        """
        if sample_type == 'in':
            return [pd.read_pickle("./{}.pkl".format(self.FL_IN[0]))]
        elif sample_type == 'out':
            return [pd.read_pickle("./{}.pkl".format(self.FL_OUT[0]))]

    @property
    def get_data(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Get whole data.
        """
        datas = list(itertools.chain.from_iterable(self.__dict__.values()))
        return [pd.read_pickle("./{}.pkl".format(data)) for data in datas]


if __name__ == "__main__":
    data_loader = DataLoader()
