import os
from ..application_logger.custom_logger import CustomApplicationLogger
import zipfile
import pandas as pd


class DataIngestion:
    def __init__(self) -> None:
        self.file_object = open(
            r"C:\Users\Dasitha\Downloads\Malicious-Portable-Executable-Files-Detection-using-Machine-Learning\logs\DataIngestion_logs.txt",
            "a+",
        )
        self.logger = CustomApplicationLogger()

  

    def read_data(self):
        try:
            class1 = pd.read_csv(r"C:\Users\User\Desktop\pro\data\bengin.csv")
            class2 = pd.read_csv(r"C:\Users\User\Desktop\pro\data\malware.csv")
            df = pd.concat([class1, class2])
            df = df.sample(frac=1).reset_index(drop=True)
            return df
        except Exception as e:
            raise Exception(f"Exception occured while reading data: {e}")
