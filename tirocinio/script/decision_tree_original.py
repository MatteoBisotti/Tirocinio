"""
    Questo script carica il dataset originale. Applica un modello di albero di decisione, eseguendo un metodo di holdout sui dati per la 
    divisione in training set e testing set, e una grid search con cross validation a 5 fold per 
    la selezione dei migliori iperparametri. I risultati vengono registrati in un file di log per analisi future.
"""

import pandas as pd
import logging
from sklearn.model_selection import train_test_split

import sys
sys.path.append("../base_lib")
import functions as func
import models 

logging.basicConfig(level=logging.INFO, format='%(message)s')

file_handler = logging.FileHandler('../logs/dt_original_data.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger()

logger.handlers = []
logger.addHandler(file_handler)


def main():
    dataset = pd.read_csv("../csv/dataset_original.csv")

    dataset = func.drop_cols(dataset)

    X = dataset.drop(['LUX_01'], axis=1)
    y = dataset['LUX_01']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

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