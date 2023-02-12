


import logging
logging.basicConfig(level = logging.DEBUG, filename='main_logs.log', filemode = 'w')

import pandas as pd
import numpy as np
import mlflow
import yaml

from sklearn.model_selection import  GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from data_processor import DataProcessor

# load the config
with open("config.yaml", "r") as stream:
    try:
        config_map = yaml.safe_load(stream)
        # print(config_map)
        RANDOM_STATE = config_map['RANDOM_STATE']
        data_dir = config_map['path']['data_dir']
        local_artifacts_dir = config_map['path']['local_artifacts_dir']

        dataset = config_map['path']['dataset']

        # load types of columns
        data_config = config_map['data']
        num_cols, binary_cols, cat_cols, target_col = data_config['num_cols'], data_config['binary_cols'], data_config['cat_cols'], data_config['target_col']

        # test split ratio
        TEST_SIZE = data_config['test_split_ratio']

        # model params - here I am using Random forest. But other models with params defined in config.py can be used.
        rf_params = config_map['params']['random_forest']

    except yaml.YAMLError as exc:
        print(f"Exception while reading confige file: {exc}")

if __name__ == "__main__":

    logging.info("Reading data")
    df = pd.read_csv(f'{data_dir}/{dataset}')

    data_processor = DataProcessor(num_cols, binary_cols, cat_cols, target_col, data_dir, local_artifacts_dir, mode = 'train')
    logging.info("Initiaize the data processor and set to training mode")

    train_test_split_args = {'test_size': TEST_SIZE, 'shuffle': True, 'random_state': RANDOM_STATE}
    train, test = data_processor.split_data(df, **train_test_split_args)

    with mlflow.start_run():

        logging.info("processing training data and fitting encoders ..")
        X_train, y_train = data_processor.process_data(train)

        param_grid = {'n_estimators': rf_params['n_estimators'], 'max_depth': rf_params['max_depth'], 'random_state': [RANDOM_STATE]}

        kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=RANDOM_STATE)
        clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, scoring='accuracy', cv = kfold, n_jobs = -1)
        clf.fit(X_train, y_train.values.ravel())

        # log run parameters
        mlflow.sklearn.log_model(sk_model = clf.best_estimator_, artifact_path="model")
        mlflow.log_params(clf.best_params_)

        # log the config file so its easier to check which config values were used for a particular run. Also helps in reprocing same results in the future.
        mlflow.log_artifact('config.yaml')

        # process test data using encoders fitted on train data
        # step 1 - set data processing mode to test mode. This mode uses encoders from train set to transform test data.
        data_processor.mode = 'test'

        # step 2 - process test data
        X_test, y_test = data_processor.process_data(test)

        # step 3 - get predictions on test data
        y_pred = clf.predict(X_test)

        # step 4 - calculate accuracy and log it in MLFlow for comparision
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)

        logging.info(f'\n classification report: \n {classification_report(y_test, y_pred)}')
        logging.info("Finished model training")

