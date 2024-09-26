import pandas as pd
import logging
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

import sys
sys.path.append("../base_lib")
import functions as func
import models 

logging.basicConfig(level=logging.INFO, format='%(message)s')

file_handler = logging.FileHandler('../logs/dt_dummy_data_autoencoder.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger()

logger.handlers = []
logger.addHandler(file_handler)

def filtered_feature_importance(model, feature_cols):
    importance = model.feature_importance()
        
    feature_importance = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    feature_importance = feature_importance[feature_importance['Importance'] > 0]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])

    plt.title("Importanza delle feature")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

def main():
    dataset = pd.read_csv("../csv/dataset_dummy_feature.csv")
    dataset = dataset.astype(int)

    dataset_augmented = pd.read_csv("../csv/dataset_dummy_autoencoder.csv")
    dataset_augmented = dataset_augmented.drop(['Unnamed: 0'], axis=1)
    mse = dataset_augmented['mse'][0]
    dataset_augmented = dataset_augmented.drop(['mse'], axis=1)

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
    logging.info(f"mse:{mse}")
    logging.info("-------------------------------")

    display(Markdown(f"**MSE** = {mse}"))

    feature_cols = list(X_train.columns)
    model.print_tree(feature_cols)
    filtered_feature_importance(model, feature_cols)