import pandas as pd

import sys
sys.path.append("../Imputation")
import imputation as imp 

sys.path.append("../base_lib")
import models
import functions as func

from IPython.display import display

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score

def load_csv():
    dataset = pd.read_csv("../data/datiLussazioniDefinitivi.csv", delimiter=";")

    dataset = func.clean_dataset(dataset)
    dataset = imp.total_imputation_mean_median(dataset)
    return dataset

def oversampling_SMOTENC(dataset):
    X = dataset.drop(['LUX_01'], axis=1)
    y = dataset['LUX_01']

    dataset = func.oversampling(dataset, X, y)
    return dataset

# Funzione per tracciare un grafico per ogni metrica
def plot_metric(depths, metric_values, metric_name):
    plt.figure(figsize=(12, 8))
    plt.plot(depths, metric_values, marker='o', label='Testing')
    plt.xlabel('Max Depth')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Max Depth')

    max_value = max(metric_values)

    # Traccia una linea orizzontale al massimo valore presente
    plt.axhline(max_value, color='r', linestyle='--', label=f'Max {metric_name}', alpha=0.7)

    # Aggiungi annotazione testuale sull'asse y
    plt.text(max_value + 0.2, max_value + 0.02, f'y = {max_value:.2f}', color='black', fontsize=12, ha='left', va='center')

    plt.legend()
    plt.xlim(0, 21)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xticks(np.arange(min(depths), max(depths)+1, 2))  # Imposta i valori delle ascisse a intervalli di 2
    plt.show()


def train(training_set, testing_set):
    feature_cols = ['BREED', 'GENDER_01', 'AGEATSURGERYmo', 'BODYWEIGHTKG', 'Taglia', 'BCS', 
                'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE', 'ALO', 'CUPRETROVERSION', 'STEMANTEVERSIONREAL', 
                'RECTUSFEMORISM.RELEASE', 'LUX_CR']

    X_train = training_set[feature_cols]
    y_train = training_set['LUX_01']

    X_test = testing_set[feature_cols]
    y_test = testing_set['LUX_01']

    dt = DecisionTreeClassifier(random_state=42, criterion='gini', min_impurity_decrease=0.0, min_samples_split=4)

    param_grid = {
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    }

    grid_search = GridSearchCV(estimator=dt, cv=5, param_grid=param_grid, scoring='accuracy')

    grid_search.fit(X_train, y_train)

    # Salva il valore dell'accuretezza per ogni valore di max_depth
    depths = param_grid['max_depth']
    test_accuracies = []

    for depth in depths:
        model = DecisionTreeClassifier(random_state=42, criterion='gini', min_impurity_decrease=0.0, min_samples_split=4, max_depth=depth)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        
        test_accuracies.append(accuracy_score(y_test, y_test_pred))

    plot_metric(depths, test_accuracies, 'Accuratezza')

def main():
    dataset = load_csv()
    df = oversampling_SMOTENC(dataset)

    dataset = func.drop_cols(dataset)
    df = func.drop_cols(df)

    training_set, testing_set = func.train_test(dataset, df, False)
    train(training_set, testing_set)