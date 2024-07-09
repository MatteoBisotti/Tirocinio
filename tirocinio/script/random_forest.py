import sys
sys.path.append("../Imputation")
import imputation as imp 
sys.path.append("../base_lib")
import pandas as pd
import models
import functions as func
from IPython.display import display
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('../logs/rf_model.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger()

logger.handlers = []
logger.addHandler(file_handler)

def main():
    dataset = pd.read_csv("../csv/dataset_original.csv")
    df = pd.read_csv("../csv/dataset_SMOTENC.csv")

    dataset = func.drop_cols(dataset)
    df = func.drop_cols(df)

    display(df.head(5))

    func.plot_outcome_feature(dataset, 'LUX_01')
    func.plot_outcome_feature(df, 'LUX_01')

    training_set, testing_set = func.train_test(dataset, df, False)

    feature_cols = ['BREED', 'GENDER_01', 'AGEATSURGERYmo', 'BODYWEIGHTKG', 'Taglia', 'BCS', 
                'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE', 'ALO', 'CUPRETROVERSION', 'STEMANTEVERSIONREAL', 
                'RECTUSFEMORISM.RELEASE', 'LUX_CR']

    X_train = training_set[feature_cols]
    y_train = training_set['LUX_01']

    X_test = testing_set[feature_cols]
    y_test = testing_set['LUX_01']

    param_grid = {
        'n_estimators': [3, 4, 5],
        'max_depth': [8],
        'criterion': ["gini", "entropy"],
        'min_samples_split': [2, 4, 6],
        'min_impurity_decrease': [0.0, 0.01, 0.02]
    }

    model, metrics_df = models.random_forest_gridsearchcv_model(X_train, X_test, y_train, y_test, param_grid, 5, 'f1_macro')

    logger.info(f"Metrics:\n{metrics_df}")

