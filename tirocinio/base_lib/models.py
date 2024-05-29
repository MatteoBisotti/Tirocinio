from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.logistic_regression import LogisticRegressionModel
from model.logistic_regression_cv import LogisticRegressionCvModel
from model.logistic_regression_gscv import LogisticRegressionGscvModel

from model.dt import DecisionTreeModel
from model.dt_gscv import DecisionTreeGscvModel

from model.rf import RandomForestModel
from model.rf_gscv import RandomForestGscvModel

import time

import os
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname('var.env'))
env_path = os.path.join(project_root, '.env')

# Carica le variabili d'ambiente dal file .env
load_dotenv(dotenv_path=env_path)

# Recupera il valore della variabile d'ambiente
random_state = int(os.getenv('RANDOM_STATE', 42))

# modello di regressione logistica
def logistic_regression_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.3)

    model = LogisticRegressionModel()

    start_time = time.time()
    model.train(X_train, y_train)
    end_time = time.time()

    model.print_report(X_test, y_test)

    t_time = end_time - start_time
    
    return model, t_time, model.get_report(X_test, y_test)


# modello di regressione logistica con cross validation
def logistic_regression_cv_model(X, y, cv):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    model = LogisticRegressionCvModel(cv=cv)

    start_time = time.time()
    model.train(X, y)
    end_time = time.time()

    model.print_report(X_test, y_test)

    t_time = end_time - start_time
    
    return model, t_time, model.get_report(X_test, y_test)


# modello di regressione logistica con grid search cv
def logistic_regression_gridsearchcv_model(X, y, param_grid, cv, scoring):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    model = LogisticRegressionGscvModel(param_grid=param_grid, cv=cv, scoring=scoring)

    start_time = time.time()
    model.train(X, y)
    end_time = time.time()

    model.print_report(X_test, y_test)

    print("Migliori parametri:", model.print_best_params())

    t_time = end_time - start_time
    
    return model, t_time, model.get_report(X_test, y_test)



# modello di albero di decisione
def decision_tree_model(X, y, max_depth):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.3)

    model = DecisionTreeModel(max_depth=max_depth)

    start_time = time.time()
    model.train(X_train, y_train)
    end_time = time.time()

    model.print_report(X_test, y_test)

    t_time = end_time - start_time
    
    return model, t_time, model.get_report(X_test, y_test)


# modello di albero di decisione con grid search cv
def decision_tree_gridsearchcv_model(X, y, param_grid, cv, scoring):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.3)

    model = DecisionTreeGscvModel(param_grid=param_grid, cv=cv, scoring=scoring)

    start_time = time.time()
    model.train(X_train, y_train)
    end_time = time.time()

    model.print_report(X_test, y_test)

    print("Migliori parametri:", model.print_best_params())

    t_time = end_time - start_time
    
    return model, t_time, model.get_report(X_test, y_test)


# modello di random forest
def random_forest_model(X, y, n_estimators, max_depth):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    model = RandomForestModel(n_estimators=n_estimators, max_depth=max_depth)

    start_time = time.time()
    model.train(X_train, y_train)
    end_time = time.time()

    model.print_report(X_test, y_test)

    t_time = end_time - start_time
    
    return model, t_time, model.get_report(X_test, y_test)


# modello di random forest con grid search cv
def random_forest_gridsearchcv_model(X, y, param_grid, cv, scoring):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    model = RandomForestGscvModel(param_grid=param_grid, cv=cv, scoring=scoring)

    start_time = time.time()
    model.train(X_train, y_train)
    end_time = time.time()

    model.print_report(X_test, y_test)

    print("Migliori parametri:", model.print_best_params())
    
    t_time = end_time - start_time
    
    return model, t_time, model.get_report(X_test, y_test)


def feature_importance_conf(importance, importance_grid, feature_cols):
    # Larghezza delle barre
    bar_width = 0.35

    # Posizioni sull'asse y per le due serie
    y_pos1 = np.arange(len(feature_cols))
    y_pos2 = np.arange(len(feature_cols)) + bar_width

    # Calcolare le posizioni delle barre in modo che siano centrate rispetto alle etichette
    centered_y_pos1 = y_pos1 - bar_width / 2
    centered_y_pos2 = y_pos2 - bar_width / 2

    # Visualizza l'importanza delle feature in un unico grafico a barre
    plt.figure(figsize=(12, 6))

    plt.barh(centered_y_pos1, importance_grid, bar_width, align='center', label='Albero di decisione con grid search CV')
    plt.barh(centered_y_pos2, importance, bar_width, align='center', label='Albero di decisione')

    plt.yticks(y_pos1, feature_cols)
    plt.xlabel('Importanza delle feature')
    plt.ylabel('Feature')
    plt.title('Confronto dell\'importanza dell\'importanza delle feature tra albero di decisione e albero di decisione con grid search CV')
    plt.legend()

    plt.show()