
import mlflow
import json
import pandas as pd
from data_processor import DataProcessor

DATA_DIR = './data'

df = mlflow.search_runs(experiment_ids=["0"], order_by=["metrics.accuracy"])


obj = df[(df['status'] == 'FINISHED')].sort_values(by = ['metrics.accuracy', "end_time"], ascending=False).iloc[0, :]['tags.mlflow.log-model.history']

best_run_id = json.loads(obj)[0]['run_id']

# load mlflow model with run_id
logged_model = f'runs:/{best_run_id}/model'
env = mlflow.pyfunc.get_model_dependencies(logged_model)
print(env)
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# here I will just import test df since there is no real inference (prod) data in this project 
inference_df = pd.read_csv('./data/test.csv').sample(5)

num_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
binary_cols = []
categorical_cols = ['color', 'clarity']
target_col = ['cut']

data_processor = DataProcessor(num_cols, binary_cols, categorical_cols, target_col, mode = 'test')

# extract only the features 
X_inference, _ = data_processor.process_data(inference_df)

# Predict on a Pandas DataFrame.
predictions = loaded_model.predict(X_inference)

X_inference['predictions'] = predictions

# store predictions in the data folder
X_inference.to_csv(f'{DATA_DIR}/inference_predictions.csv', index=False)

