import pandas as pd 
import numpy as np
from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split

def load_object(filepath):
    try:
        return pd.read_csv(filepath)
        logging.info(f"data is read ")
    except FileNotFoundError:
        print(f"file not found in {filepath}")
        return None
