from typing import Union
from pathlib import Path
from os.path import isdir
from warnings import warn

import pandas as pd
import csv

class PatientsCSVLoader:

    @staticmethod
    def load_dataframe(path: Union[Path, str]) -> Union[None, pd.DataFrame]:
        """ Loads the patient data from the csv file at the given path, into a pandas dataframe.

        :param path: the data path
        :return: a pandas dataframe
        """
        if not isdir(path):
            warn(f"The given path does not exist: {path}.")
            return None
        df = pd.read_csv(path)
        return df

    @staticmethod
    def load_dict(path: Union[Path, str]) -> Union[None, dict]:
        """ Loads the patient data from the csv file at the given path, into a python dictionary.

        :param path: the data path
        :return: a python dictionary
        """

        if not isdir(path):
            warn(f"The given path does not exist: {path}.")
            return None
        keys = ("user_id", "night_id", "age", "sex", "height",
                "weight", "pulse", "BPsys/BPdia", "ODI", "NAp", "NHyp", "AI", "HI", "AHI")

        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            # INCOMPLETE !
            pass