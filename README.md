# mlflow

# Goal
This repo is aimed to showcase basic capabilities of MLFlow like - building trackable experiments and to make them reproducable on any environment. The next version of it will cover model deployment and model lifecycle management. 

In this project, our aim is not to focus much on data analysis (EDA) or feature engineering but rather trying out different techniques, models, and hyperparameter tuning. In the end, we find the best performing model and show how we can use it for inference (prod data)

## Dataset
The dataset used here is kaggle diamonds dataset. Link - https://www.kaggle.com/datasets/shivam2503/diamonds
Given a set of features, our aim is to predict the quality of dimanond cut. It is a multiclass classification problem.

## Setup
For setup, having conda installed should siffuce. You can create a new conda env or one would be created automatically upon running "mlflow run" by referring to conda.yaml

## Overall steps

1. Data processing 
2. Performing model training 
3. Prediction on Test data
4. Logging the performance of model on test data
5. Steps 2, 3, 4 can be repeated by performing an experiment. Each experiment consists of multiple trials (runs) which involves modifying hyperparameters, models, encoders, etc. All of the parameters/artifacts can be logged under MLFlow. 
6. choose the best model (found by a unique run_id) in the experiment which will be used for Inference predictions.
7. To be implemented - Deploy the best model using MLFlow serving for batch predictions or real-time prediction using an API.

## Execution

1. setup config.py with the required parameters. 
2. Run "mlflow run -e main.py --run-name <run_name> ./". You can use "mlflow run --help" to find other flags than can be used. 
3. Repeat steps 1 and 3 by modifying params, models, encoders, etc. 
4. Run Inference.py - this finds the model with the best accuracy and uses it for predictions on Inference data (here I just again use test data since prod real time data isn't available)
5. Inference predictions are posted in the data folder. Additionally you will see a mlruns folder that has details about yoour experiment logs. 
6. Run "mlflow ui" to start a UI that shows all your results on a browser. 
![image](https://user-images.githubusercontent.com/24752688/218344345-e293c368-6615-4ae5-913b-6862f23f4929.png)
![image](https://user-images.githubusercontent.com/24752688/218344475-8a900410-8607-4e11-833f-2b24165991de.png)



## Future development

1. Model deployment (Batch and Real-time predictions)
2. Model lifecycle management (Having model registry with versions like staging, prod, etc)
