"""
Modulo per la gestione di modelli di apprendimento automatico con Decision Tree e Random Forest.

Questo modulo include classi e funzioni per la creazione, l'addestramento e la valutazione di modelli
di albero di decisione e foresta casuale, inclusa l'ottimizzazione tramite Grid Search CV.

Funzioni:
    - DecisionTreeModel: Classe per il modello di albero di decisione.
    - DecisionTreeGscvModel: Classe per il modello di albero di decisione con Grid Search CV.
    - RandomForestModel: Classe per il modello di foresta casuale.
    - RandomForestGscvModel: Classe per il modello di foresta casuale con Grid Search CV.

Funzioni:
    - decision_tree_model: Crea e addestra un modello di albero di decisione.
    - decision_tree_gridsearchcv_model: Crea e addestra un modello di albero di decisione con Grid Search CV.
    - random_forest_model: Crea e addestra un modello di foresta casuale.
    - random_forest_gridsearchcv_model: Crea e addestra un modello di foresta casuale con Grid Search CV.

Moduli esterni richiesti:
    - os
    - dotenv
    - seaborn
    - IPython.display
    - model.dt (DecisionTreeModel)
    - model.dt_gscv (DecisionTreeGscvModel)
    - model.rf (RandomForestModel)
    - model.rf_gscv (RandomForestGscvModel)
"""

import os
from dotenv import load_dotenv
import seaborn as sns
from IPython.display import display
from model.dt import DecisionTreeModel
from model.dt_gscv import DecisionTreeGscvModel
from model.rf import RandomForestModel
from model.rf_gscv import RandomForestGscvModel

# Imposta il percorso del file .env
project_root = os.path.dirname(os.path.dirname('var.env'))
env_path = os.path.join(project_root, '.env')

# Carica le variabili d'ambiente dal file .env
load_dotenv(dotenv_path=env_path)

# Recupera il valore della variabile d'ambiente
random_state = int(os.getenv('RANDOM_STATE', 42))

# Colonne delle caratteristiche
feature_cols = ['BREED', 'GENDER_01', 'AGEATSURGERYmo', 'BODYWEIGHTKG', 'Taglia', 'BCS', 
                'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE', 'ALO', 'CUPRETROVERSION', 
                'STEMANTEVERSIONREAL', 'RECTUSFEMORISM.RELEASE', 'LUX_CR']

def decision_tree_model(X_train, X_test, y_train, y_test, max_depth, min_sample_split, min_impurity_decrease, criterion):
    """
    Crea e addestra un modello di albero di decisione.

    Args:
        X_train (DataFrame): Dati di addestramento delle caratteristiche.
        X_test (DataFrame): Dati di test delle caratteristiche.
        y_train (Series): Dati di addestramento delle etichette.
        y_test (Series): Dati di test delle etichette.
        max_depth (int): Profondità massima dell'albero.
        min_sample_split (int): Numero minimo di campioni richiesti per dividere un nodo interno.
        min_impurity_decrease (float): Riduzione minima dell'impurità per effettuare una divisione.
        criterion (str): Criterio per misurare la qualità di una divisione (es. "gini" o "entropy").

    Returns:
        DecisionTreeModel: Modello addestrato di albero di decisione.
    """
    model = DecisionTreeModel(max_depth=max_depth, 
                              min_sample_split=min_sample_split, 
                              min_impurity_decrease=min_impurity_decrease, 
                              criterion=criterion)
    
    model.train(X_train, y_train)

    metrics_df = model.statistics(X_test, y_test)
    
    return model, metrics_df

def decision_tree_gridsearchcv_model(X_train, X_test, y_train, y_test, param_grid, cv, scoring):
    """
    Crea e addestra un modello di albero di decisione con Grid Search CV.

    Args:
        X_train: Dati di addestramento delle caratteristiche.
        X_test: Dati di test delle caratteristiche.
        y_train: Dati di addestramento delle etichette.
        y_test: Dati di test delle etichette.
        param_grid: Dizionario di parametri per la ricerca a griglia.
        cv: Numero di fold per la validazione incrociata.
        scoring: Metodologia di scoring per la validazione incrociata.

    Returns:
        tuple: Modello addestrato di albero di decisione con Grid Search CV e report dei risultati.
    """
    model = DecisionTreeGscvModel(param_grid=param_grid, cv=cv, scoring=scoring)

    model.train(X_train, y_train)

    results = model.get_best_params()
    display(results)

    metrics_df = model.statistics(X_test, y_test)
    
    return model, results, metrics_df

def random_forest_model(X_train, X_test, y_train, y_test, n_estimators, max_depth, feature):
    """
    Crea e addestra un modello di foresta casuale.

    Args:
        X_train (DataFrame): Dati di addestramento delle caratteristiche.
        X_test (DataFrame): Dati di test delle caratteristiche.
        y_train (Series): Dati di addestramento delle etichette.
        y_test (Series): Dati di test delle etichette.
        n_estimators (int): Numero di alberi nella foresta.
        max_depth (int): Profondità massima degli alberi.

    Returns:
        tuple: Modello addestrato di foresta casuale e report dei risultati.
    """
    model = RandomForestModel(n_estimators=n_estimators, max_depth=max_depth)

    model.train(X_train, y_train)

    metrics_df = model.statistics(X_test, y_test)
    display(metrics_df)
    model.plot_metrics(metrics_df)
    
    model.print_tree(feature_cols)
    model.graph_feature_importance(feature_cols)
    
    return model, model.get_report(X_test, y_test)

def random_forest_gridsearchcv_model(X_train, X_test, y_train, y_test, param_grid, cv, scoring):
    """
    Crea e addestra un modello di foresta casuale con Grid Search CV.

    Args:
        X_train (DataFrame): Dati di addestramento delle caratteristiche.
        X_test (DataFrame): Dati di test delle caratteristiche.
        y_train (Series): Dati di addestramento delle etichette.
        y_test (Series): Dati di test delle etichette.
        param_grid (dict): Dizionario di parametri per la ricerca a griglia.
        cv (int): Numero di fold per la validazione incrociata.
        scoring (str): Metodologia di scoring per la validazione incrociata.

    Returns:
        model: Modello addestrato di foresta casuale con Grid Search CV, metrica dei risultati e migliori parametri
        results: Migliori parametri della grid search
        metrics_df
    """
    model = RandomForestGscvModel(param_grid=param_grid, cv=cv, scoring=scoring)

    model.train(X_train, y_train)

    results = model.get_best_params()
    display(results)

    metrics_df = model.statistics(X_test, y_test)
    
    return model, results, metrics_df

def random_forest_grid_search_pca(X_train, X_test, y_train, y_test, param_grid, cv, scoring):
    model = RandomForestGscvModel(param_grid=param_grid, cv=cv, scoring=scoring)

    model.train(X_train, y_train)

    results = model.get_best_params()
    display(results)

    metrics_df = model.statistics(X_test, y_test)
    
    return model, results, metrics_df