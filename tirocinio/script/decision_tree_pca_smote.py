import pandas as pd
import logging

import sys
sys.path.append("../base_lib")
import functions as func
import models 

logging.basicConfig(level=logging.INFO, format='%(message)s')

file_handler = logging.FileHandler('../logs/dt_pca_data_smotenc.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger()

logger.handlers = []
logger.addHandler(file_handler)

def main():
    dataset = pd.read_csv("../csv/dataset_pca.csv")

    X = dataset.drop(['LUX_01'], axis=1)
    y = dataset['LUX_01']
    dataset_augmented = func.oversampling_SMOTE(dataset, X, y)
    training_set, testing_set = func.train_test(dataset, dataset_augmented, False)

    X_train = training_set.drop(['LUX_01'], axis=1)
    y_train = training_set['LUX_01']
    X_test = testing_set.drop(['LUX_01'], axis=1)
    y_test = testing_set['LUX_01']

    param_grid = {
        'max_depth': [8],
        'criterion': ["gini", "entropy"],
        'min_samples_split': [2, 4, 6],
        'min_impurity_decrease': [0.0, 0.01, 0.02]
    }

    model, results, metrics_df = models.decision_tree_gridsearchcv_model(X_train, X_test, 
                                                                             y_train, y_test, 
                                                                             param_grid=param_grid, 
                                                                             cv=5, 
                                                                             scoring='f1_macro')
    
    logging.info(f"criterio:{results['criterion'][0]}:max_depth:{results['max_depth'][0]}:min_impurity_decrease:{results['min_impurity_decrease'][0]}:min_samples_split:{results['min_samples_split'][0]}")
    logging.info(f"accuracy:{metrics_df['Valore'][0]}:precision:{metrics_df['Valore'][1]}:recall:{metrics_df['Valore'][2]}:f1_score:{metrics_df['Valore'][3]}:roc_auc:{metrics_df['Valore'][4]}:specificity:{metrics_df['Valore'][5]}")
    logging.info("-------------------------------")