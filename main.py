


import logging
logging.basicConfig(level = logging.DEBUG, filename='main_logs.log', filemode = 'w')

import pandas as pd
import numpy as np
import mlflow

from sklearn.model_selection import  GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from data_processor import DataProcessor

filename = 'diamonds.csv'
DATA_DIR = './data'
RANDOM_STATE = 40

if __name__ == "__main__":

    logging.info("Reading data")
    df = pd.read_csv(f'{DATA_DIR}/{filename}')

    num_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
    binary_cols = []
    categorical_cols = ['color', 'clarity']
    target_col = ['cut']

    data_processor = DataProcessor(num_cols, binary_cols, categorical_cols, target_col, mode = 'train')
    logging.info("Initiaize the data processor and set to training mode")

    train_test_split_args = {'test_size': 0.1, 'shuffle': True, 'random_state': RANDOM_STATE}
    train, test = data_processor.split_data(df, **train_test_split_args)

    with mlflow.start_run():

        logging.info("processing training data and fitting encoders ..")
        X_train, y_train = data_processor.process_data(train)

        param_grid = {'n_estimators': [20, 30], 'max_depth': [5, 8, 10]}

        clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, scoring='accuracy', cv = 5, n_jobs = -1)
        clf.fit(X_train, y_train.values.ravel())

        # log run parameters
        mlflow.sklearn.log_model(clf.best_estimator_, "model")
        mlflow.log_params(clf.best_params_)

        # process test data using encoders fitted on train data
        # step 1 - set data processing mode to test
        data_processor.mode = 'test'

        # step 2 - process test data
        X_test, y_test = data_processor.process_data(test)

        # get predictions on test data
        y_pred = clf.predict(X_test)

        # calculate accuracy and log it in MLFlow for comparision
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)

        # should i register the model?


        logging.info(f'\n classification report: \n {classification_report(y_test, y_pred)}')
        logging.info("Finished model training")

