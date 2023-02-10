import os
import joblib
import logging
logging.basicConfig(level = logging.DEBUG)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import mlflow
import numpy as np

from utilities import save_pkl, load_pkl


class DataProcessor():
    """
    Class to process input data to make it suitable for model training. 
    """
    def __init__(self, num_cols, binary_cols, categorical_cols, target_col, mode):
        self.mode = mode
        self.num_cols = num_cols
        self.binary_cols = binary_cols
        self.cat_cols = categorical_cols
        self.target_col = target_col
        
        self.DATA_DIR = './data'
        self.MODEL_DIR = './model'
        
        self.encoders = {}
    
    def split_data(self, df, **kwargs):
        """
        splits a given dataframe into train and test.

        Uses sklearn train_test_split function. Also saves the dataframe in the DATA_DIR for future use. 

        Args:
            df: pandas dataframe
            **kwargs: keyword arguments to send to train test split function

        Returns:
            train, test: Train and Test dataframes
        """
        self.train, self.test = train_test_split(df, **kwargs)

        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR, exist_ok=True)

        logging.info(f'train data rows: {self.train.shape[0]}, test data rows: {self.test.shape[0]}')
        
        self.train.to_csv(self.DATA_DIR + '/train.csv')
        self.test.to_csv(self.DATA_DIR + '/test.csv')
        
        return (self.train, self.test)
    
    
    def process_data(self, df):
        """
        processes a Dataframe using various encoders and scaler if required.

        Encodes input columns into suitable format for training. For Eg, using
        OHE, Label Encoder for categorical features or scaler for numerical features.
        It has 2 modes (set using object.mode):
             i. Train mode - Fit the encoder and transform the data.Also saves the encoder as .pkl file.
             ii. Test mode - Load the encoder stored as .pkl and transform the data.

        Args:
            df: pandas dataframe to be processed. 

        Returns:
            X: Input training features for the model.
            y: Target label

        """
        
        if(self.mode == 'train'):
            
            # ordinally encode the categorical cols
            for col in self.cat_cols:
                enc = OrdinalEncoder()
                enc.fit(np.array(df[col]).reshape(-1, 1))
                
                self.encoders[col] = enc
            
            save_pkl(self.encoders, dir = self.MODEL_DIR,   filename = 'encoder.pkl')
            save_pkl(df.columns.tolist(), dir = self.MODEL_DIR,   filename = 'train_columns.pkl')

            mlflow.log_artifact(local_path=self.MODEL_DIR + '/encoder.pkl')
            mlflow.log_artifact(local_path=self.MODEL_DIR + '/train_columns.pkl')
            # save column
            
            self.columns = df.columns.tolist()
        
        elif (self.mode == 'test'):
            # load encoders
            self.encoders = load_pkl(dir = self.MODEL_DIR, filename = 'encoder.pkl')
            self.columns = load_pkl(dir = self.MODEL_DIR, filename = 'train_columns.pkl')
            
            
        for col in self.cat_cols:
            enc = self.encoders[col]
            
            df.loc[:, col] = enc.transform(np.array(df[col]).reshape(-1, 1))
            
        df = df[self.columns]
        
        self.y, self.X = df[self.target_col], df.drop(self.target_col, axis=1)
        
        return self.X, self.y
    
    
    