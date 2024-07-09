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

import functions as func 

from imblearn.over_sampling import SMOTENC

import seaborn as sns

from IPython.display import display

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

project_root = os.path.dirname(os.path.dirname('var.env'))
env_path = os.path.join(project_root, '.env')

# Carica le variabili d'ambiente dal file .env
load_dotenv(dotenv_path=env_path)

# Recupera il valore della variabile d'ambiente
random_state = int(os.getenv('RANDOM_STATE', 42))

feature_cols = ['BREED', 'GENDER_01', 'AGEATSURGERYmo', 'BODYWEIGHTKG', 'Taglia', 'BCS', 
                'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE', 'ALO', 'CUPRETROVERSION', 'STEMANTEVERSIONREAL', 
                'RECTUSFEMORISM.RELEASE', 'LUX_CR']

# modello di regressione logistica
def logistic_regression_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.25)

    model = LogisticRegressionModel()

    start_time = time.time()
    model.train(X_train, y_train)
    end_time = time.time()

    model.print_report(X_test, y_test)

    model.statistics(X_test, y_test)

    t_time = end_time - start_time
    
    return model, t_time, model.get_report(X_test, y_test)


# modello di regressione logistica con cross validation
def logistic_regression_cv_model(X, y, cv):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    model = LogisticRegressionCvModel(cv=cv)

    start_time = time.time()
    model.train(X, y)
    end_time = time.time()

    model.print_report(X_test, y_test)

    model.statistics(X_test, y_test)

    t_time = end_time - start_time
    
    return model, t_time, model.get_report(X_test, y_test)


# modello di regressione logistica con grid search cv
def logistic_regression_gridsearchcv_model(X_train, X_test, y_train, y_test, param_grid, cv, scoring):

    model = LogisticRegressionGscvModel(param_grid=param_grid, cv=cv, scoring=scoring)

    model.train(X_train, y_train)

    metrics_df = model.statistics(X_test, y_test)
    display(metrics_df)
    model.plot_metrics(metrics_df)
    
    return model, model.get_report(X_test, y_test)


# modello di albero di decisione
def decision_tree_model(X_train, X_test, y_train, y_test, max_depth, min_sample_split, min_impurity_decrease, criterion):

    model = DecisionTreeModel(max_depth=max_depth, 
                              min_sample_split=min_sample_split, 
                              min_impurity_decrease=min_impurity_decrease, 
                              criterion=criterion)
    
    model.train(X_train, y_train)
    
    return model

# modello di albero di decisione con grid search cv
def decision_tree_gridsearchcv_model(X_train, X_test, y_train, y_test, param_grid, cv, scoring):

    model = DecisionTreeGscvModel(param_grid=param_grid, cv=cv, scoring=scoring)

    model.train(X_train, y_train)

    results = model.get_best_params()
    display(results)

    metrics_df = model.statistics(X_test, y_test)
    display(metrics_df)
    model.plot_metrics(metrics_df)
    
    model.print_tree(feature_cols)
    model.graph_feature_importance(feature_cols)
    
    return model, model.get_report(X_test, y_test)


# modello di random forest
def random_forest_model(X_train, X_test, y_train, y_test, n_estimators, max_depth):
    model = RandomForestModel(n_estimators=n_estimators, max_depth=max_depth)

    model.train(X_train, y_train)

    metrics_df = model.statistics(X_test, y_test)
    display(metrics_df)
    model.plot_metrics(metrics_df)
    
    model.print_tree(feature_cols)
    model.graph_feature_importance(feature_cols)
    
    return model, model.get_report(X_test, y_test)


# modello di random forest con grid search cv
def random_forest_gridsearchcv_model(X_train, X_test, y_train, y_test, param_grid, cv, scoring):

    model = RandomForestGscvModel(param_grid=param_grid, cv=cv, scoring=scoring)

    model.train(X_train, y_train)

    results = model.get_best_params()
    display(results)

    metrics_df = model.statistics(X_test, y_test)
    display(metrics_df)
    model.plot_metrics(metrics_df)
    
    model.print_tree(feature_cols)
    model.graph_feature_importance(feature_cols)
    
    return model, metrics_df

