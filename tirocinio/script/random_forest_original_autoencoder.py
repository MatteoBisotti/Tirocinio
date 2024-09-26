import pandas as pd
import logging
from IPython.display import display, Markdown

import sys
sys.path.append("../base_lib")
import functions as func
import models 

logging.basicConfig(level=logging.INFO, format='%(message)s')

file_handler = logging.FileHandler('../logs/rf_original_data_autoencoder.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger()

logger.handlers = []
logger.addHandler(file_handler)

def main():
    dataset = pd.read_csv("../csv/dataset_original.csv")
    dataset = func.drop_cols(dataset)

    dataset_augmented = pd.read_csv("../csv/dataset_original_autoencoder.csv")
    dataset_augmented = dataset_augmented.drop(['Unnamed: 0'], axis=1)
    mse = dataset_augmented['mse'][0]
    dataset_augmented = dataset_augmented.drop(['mse'], axis=1)

    trainig_set, testing_set = func.train_test(dataset, dataset_augmented, False)
    X_train = trainig_set.drop('LUX_01', axis=1)
    y_train = trainig_set['LUX_01']
    X_test = testing_set.drop('LUX_01', axis=1)
    y_test = testing_set['LUX_01']

    param_grid = {
        'n_estimators': [3, 4, 5],
        'max_depth': [8],
        'criterion': ["gini", "entropy"],
        'min_samples_split': [2, 4, 6],
        'min_impurity_decrease': [0.0, 0.01, 0.02]
    }

    model, results, metrics_df = models.random_forest_gridsearchcv_model(X_train, X_test, 
                                                                             y_train, y_test, 
                                                                             param_grid=param_grid, 
                                                                             cv=5, 
                                                                             scoring='f1_macro')
    
    logging.info(f"criterio:{results['criterion'][0]}:max_depth:{results['max_depth'][0]}:min_impurity_decrease:{results['min_impurity_decrease'][0]}:min_samples_split:{results['min_samples_split'][0]}:n_estimators:{results['n_estimators'][0]}")
    logging.info(f"accuracy:{metrics_df['Valore'][0]}:precision:{metrics_df['Valore'][1]}:recall:{metrics_df['Valore'][2]}:f1_score:{metrics_df['Valore'][3]}:roc_auc:{metrics_df['Valore'][4]}:specificity:{metrics_df['Valore'][5]}")
    logging.info(f"mse:{mse}")
    logging.info("-------------------------------")

    display(Markdown(f"**MSE dell'autoencoder:** {mse:.4f}"))

    feature_cols = list(X_train.columns)
    model.print_tree(feature_cols)
    model.graph_feature_importance(feature_cols)