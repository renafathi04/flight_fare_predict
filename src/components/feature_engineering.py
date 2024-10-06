import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


class FeatureEngineering:
    def __init__(self,df):
        self.df=df

    def drop_unwanted_cols(self,unwanted_cols):
        self.df=self.df.drop(columns=unwanted_cols)
        logging.info(f"unwanted columns dropped")
        return self
     
    def handle_missing_value(self,cols,strategy='mean'):
        imputer=SimpleImputer(strategy=strategy)
        self.df[cols]=imputer.fit_transform(self.df[cols])
        logging.info(f"handled missing values")
        return self
    
    
    def drop_duplicate(self):
        self.df=self.df.drop_duplicates()
        logging.info(f"removed duplicates ")
        return self
    
    def convert_date(self,date_columns):
        for col in date_columns:
            self.df[col]=pd.to_datetime(self.df[col],dayfirst=True, errors='coerce')
            self.df[col + '__month']=self.df[col].dt.month
            self.df[col + '__day']=self.df[col].dt.day
            self.df=self.df.drop(columns=[col])
        return self
    
    def convert_time(self,time_columns):
        for col in time_columns:
            self.df[col]=pd.to_datetime(self.df[col],errors='coerce')
            self.df[col + '__hour']=self.df[col].dt.hour
            self.df[col + '__minute']=self.df[col].dt.minute
            self.df=self.df.drop(columns=[col])
        return self
    
    def convert_duration(self, duration_column):
    
    
        def extract_duration(duration):
            try:
            # Remove any extra spaces
                duration = duration.strip()

            # Initialize hours and minutes
                hours = 0
                minutes = 0
            
            # Split based on spaces and parse
                parts = duration.split()
                for part in parts:
                    if 'h' in part:
                       hours = int(part.replace('h', '').strip())
                    elif 'm' in part:
                       minutes = int(part.replace('m', '').strip())
            
                return hours, minutes
            except Exception as e:
                logging.warning(f"Could not parse duration '{duration}'. Error: {e}")
                return 0, 0  # Default to 0 hours and 0 minutes if parsing fails

    
        self.df['Duration_hours'], self.df['Duration_minutes'] = zip(*self.df[duration_column].apply(extract_duration))
    
    
        self.df = self.df.drop(columns=[duration_column])
        logging.info("Converted Duration column into hours and minutes.")
    
        return self
    
    def label_encoding(self,label_columns):
        le =LabelEncoder()
        for col in label_columns:
            self.df[col]=le.fit_transform(self.df[col])
        return self
    
    def one_hot_encoding(self,specific_cols):
        self.df=pd.get_dummies(self.df,columns=specific_cols)
        return self
    
    def get_processed_data(self):

        return self.df
