import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.datasets.dataset_base import DatasetBase
from src.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
)

class Sap(DatasetBase):
    def __init__(self):
        """SAP

        SAP dataset.
        """
        super().__init__("sap", url='datasets/SAP')
        
    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        file_name = f"{self.dataset_dir}/raw/sap_interaction.txt"

        data = pd.read_csv(
            file_name,
            header=None,
            sep="\t",
            names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL],
        )
        
        data[DEFAULT_RATING_COL] = np.ones(len(data))
        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )
        
    def load_split(self, config):
        
        df = pd.read_table(f"{self.dataset_dir}/raw/sap_interaction.txt", sep='\t', names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
        df[DEFAULT_RATING_COL] = np.ones(len(df))
        
        train, temp = train_test_split(df, test_size=0.2)
        valid, test = train_test_split(temp, test_size=0.5)
        
        print(train)
        print('-----------')
        print(valid)
        print('-----------')
        print(test)
        print('-----------')
        
        return train, valid, test