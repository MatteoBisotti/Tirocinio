"""
Modulo per la gestione di modelli di apprendimento supervisionato: Decision Tree e Random Forest.

Questo modulo include classi e funzioni per la creazione, l'addestramento e la valutazione di modelli
di albero di decisione e random forest, inclusa l'ottimizzazione degli iperparametri tramite Grid Search CV.
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

"""
# Colonne utilizzate come predittor
feature_cols = ['BREED', 'GENDER_01', 'AGEATSURGERYmo', 'BODYWEIGHTKG', 'Taglia', 'BCS', 
                'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE', 'ALO', 'CUPRETROVERSION', 
                'STEMANTEVERSIONREAL', 'RECTUSFEMORISM.RELEASE', 'LUX_CR']
"""

def decision_tree_gridsearchcv_model(X_train, X_test, y_train, y_test, param_grid, cv, scoring):
    """
    Crea e addestra un modello di albero di decisione con Grid Search CV.

    Returns:
        tuple: Modello addestrato di albero di decisione con Grid Search CV e report dei risultati.
    """
    model = DecisionTreeGscvModel(param_grid=param_grid, cv=cv, scoring=scoring)

    model.train(X_train, y_train)

    results = model.get_best_params()
    display(results)

    metrics_df = model.statistics(X_test, y_test)
    
    return model, results, metrics_df

def random_forest_gridsearchcv_model(X_train, X_test, y_train, y_test, param_grid, cv, scoring):
    """
    Crea e addestra un modello di random forest con Grid Search CV.

    Returns:
        model: Modello addestrato di foresta casuale con Grid Search CV, metrica dei risultati e migliori parametri
        results: Migliori parametri della grid search
        metrics_df: Un dataframe con i valori delle metriche
    """
    model = RandomForestGscvModel(param_grid=param_grid, cv=cv, scoring=scoring)

    model.train(X_train, y_train)

    results = model.get_best_params()
    display(results)

    metrics_df = model.statistics(X_test, y_test)
    
    return model, results, metrics_df