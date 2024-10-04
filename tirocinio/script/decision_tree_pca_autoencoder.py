import pandas as pd
import logging
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

import sys
sys.path.append("../base_lib")
import functions as func
import models 

from model.dt_gscv import DecisionTreeGscvModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(message)s')

file_handler = logging.FileHandler('../logs/dt_pca_data_autoencoder.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger()

logger.handlers = []
logger.addHandler(file_handler)


def main():

    dataset = pd.read_csv("../csv/dataset_pca.csv")

    dataset_augmented = pd.read_csv("../csv/dataset_pca_autoencoder.csv")
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
    
    model = DecisionTreeGscvModel(param_grid=param_grid, cv=5, scoring='f1_macro')

    model.train(X_train, y_train)

    results = model.get_best_params()
    display(results)

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()  

    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions, zero_division=1, average='macro')
    precision = precision_score(y_test, predictions, zero_division=1, average='macro')
    f1 = f1_score(y_test, predictions, zero_division=1, average='macro')
    roc_auc = roc_auc_score(y_test, predictions)
    specificity = tn / (tn + fp)

    data = {
        'Metrica': ['Accuracy', 'Recall', 'Precision', 'F1-score', 'ROC AUC', 'Specificity'],
        'Valore': [accuracy, recall, precision, f1, roc_auc, specificity]
    }
    metrics_df = pd.DataFrame(data)
    display(metrics_df)
    model.plot_metrics(metrics_df)

    logging.info(f"criterio:{results['criterion'][0]}:max_depth:{results['max_depth'][0]}:min_impurity_decrease:{results['min_impurity_decrease'][0]}:min_samples_split:{results['min_samples_split'][0]}")
    logging.info(f"accuracy:{metrics_df['Valore'][0]}:precision:{metrics_df['Valore'][1]}:recall:{metrics_df['Valore'][2]}:f1_score:{metrics_df['Valore'][3]}:roc_auc:{metrics_df['Valore'][4]}:specificity:{metrics_df['Valore'][5]}")
    logging.info(f"mse:{mse}")
    logging.info("-------------------------------")

    display(Markdown(f"**MSE** = {mse}"))
