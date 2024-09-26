"""
Questo modulo esegue l'elaborazione dei dati, l'addestramento del modello e la valutazione utilizzando modelli ad albero decisionale
su un dataset. Include funzioni per la visualizzazione dei dati, il calcolo delle metriche e la rappresentazione grafica
dei risultati.

Funzioni:
    load_csv(): Carica il dataset da un file CSV, lo pulisce e imputa i valori mancanti.
    oversampling_SMOTENC(dataset): Esegue l'oversampling sul dataset utilizzando SMOTENC.
    plot_metric(depths, metric_values, metric_name): Genera e visualizza un grafico per una metrica specifica in funzione della profondità massima dell'albero decisionale.
    train(training_set, testing_set): Addestra il modello ad albero decisionale, esegue una ricerca a griglia per trovare la profondità ottimale e traccia un grafico delle accuratezze di test.
    main(): Funzione principale per eseguire l'elaborazione dei dati, l'addestramento del modello e il flusso di lavoro di valutazione.

Moduli esterni richiesti:
    imp: modulo per l'imputazione dei dati.
    models: modulo contenente il modello di machine learning.
    functions: modulo contenente varie funzioni di supporto.
    pandas
    matplotlib
    seaborn
    IPython.display
    sys
    numpy
    sklearn.tree
    sklearn.model_selection
    sklearn.metrics

"""

import pandas as pd
import sys
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.append("../Imputation")
import imputation as imp 

sys.path.append("../base_lib")
import models
import functions as func

def plot_metric(depths, metric_values, metric_name):
    """
    Genera e visualizza un grafico per una metrica specifica in funzione della profondità massima dell'albero decisionale.

    Args:
        depths (list): Lista dei valori di profondità massima.
        metric_values (list): Lista dei valori della metrica corrispondente alle profondità.
        metric_name (str): Nome della metrica da visualizzare.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(depths, metric_values, marker='o', label='Testing')
    plt.xlabel('Max Depth')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Max Depth')

    max_value = max(metric_values)
    plt.axhline(max_value, color='r', linestyle='--', label=f'Max {metric_name}', alpha=0.7)
    plt.text(max_value + 0.2, max_value + 0.02, f'y = {max_value:.2f}', color='black', fontsize=12, ha='left', va='center')

    plt.legend()
    plt.xlim(0, 21)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xticks(np.arange(min(depths), max(depths)+1, 2))
    plt.show()

def train(training_set, testing_set):
    """
    Addestra il modello ad albero decisionale, esegue una ricerca a griglia per trovare la profondità ottimale e traccia un grafico delle accuratezze di test.

    Args:
        training_set (DataFrame): Il set di dati di addestramento.
        testing_set (DataFrame): Il set di dati di test.
    """
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

    depths = param_grid['max_depth']
    test_accuracies = []

    for depth in depths:
        model = DecisionTreeClassifier(random_state=42, criterion='gini', min_impurity_decrease=0.0, min_samples_split=4, max_depth=depth)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        test_accuracies.append(accuracy_score(y_test, y_test_pred))

    plot_metric(depths, test_accuracies, 'Accuratezza')

def main():
    """
    Funzione principale per eseguire l'elaborazione dei dati, l'addestramento del modello e il flusso di lavoro di valutazione.
    """
    dataset = pd.read_csv("../csv/dataset_original.csv")
    df = pd.read_csv("../csv/dataset_SMOTENC.csv")
    dataset = func.drop_cols(dataset)
    df = func.drop_cols(df)
    training_set, testing_set = func.train_test(dataset, df, False)
    train(training_set, testing_set)
