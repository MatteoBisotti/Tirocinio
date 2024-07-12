"""
Questo modulo esegue l'elaborazione dei dati, l'addestramento del modello Random Forest utilizzando GridSearchCV e la valutazione del modello
su un dataset. Include funzioni per la visualizzazione dei dati e la registrazione dei risultati tramite logging.

Funzioni:
    main(): Funzione principale per eseguire l'elaborazione dei dati, l'addestramento del modello Random Forest e la registrazione dei risultati tramite logging.

Moduli esterni richiesti:
    pandas
    logging
    re
    sys
    models: modulo contenente il modello Random Forest e la funzione per la ricerca a griglia.
    functions: modulo contenente varie funzioni di supporto.

"""

import sys
sys.path.append("../Imputation")
import imputation as imp 
sys.path.append("../base_lib")
import pandas as pd
import models
import functions as func
from IPython.display import display
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

file_handler = logging.FileHandler('../logs/rf_model_encoder.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger()

logger.handlers = []
logger.addHandler(file_handler)

def main():
    """
    Funzione principale per eseguire l'elaborazione dei dati, l'addestramento del modello Random Forest e la registrazione dei risultati tramite logging.
    
    Workflow:
        - Carica il dataset originale e il dataset con codifica.
        - Rimuove le colonne non necessarie.
        - Visualizza le prime 5 righe del dataset codificato.
        - Plotta la distribuzione dell'outcome 'LUX_01' nei due dataset.
        - Dividi il dataset in set di addestramento e test.
        - Definisce i parametri per la ricerca a griglia del modello Random Forest.
        - Addestra il modello Random Forest con GridSearchCV utilizzando metrica 'f1_macro'.
        - Registra i risultati di accuracy, precision, recall, F1-score e ROC AUC tramite logging.
    """
    dataset = pd.read_csv("../csv/dataset_original.csv")
    df = pd.read_csv("../csv/dataset_encoder.csv")

    dataset = func.drop_cols(dataset)
    df = func.drop_cols(df)

    display(df.head(5))

    func.plot_outcome_feature(dataset, 'LUX_01')
    func.plot_outcome_feature(df, 'LUX_01')

    training_set, testing_set = func.train_test(dataset, df, False)

    feature_cols = ['BREED', 'GENDER_01', 'AGEATSURGERYmo', 'BODYWEIGHTKG', 'Taglia', 'BCS', 
                'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE', 'ALO', 'CUPRETROVERSION', 'STEMANTEVERSIONREAL', 
                'RECTUSFEMORISM.RELEASE', 'LUX_CR']

    X_train = training_set[feature_cols]
    y_train = training_set['LUX_01']

    X_test = testing_set[feature_cols]
    y_test = testing_set['LUX_01']

    param_grid = {
        'n_estimators': [3, 4, 5],
        'max_depth': [8],
        'criterion': ["gini", "entropy"],
        'min_samples_split': [2, 4, 6],
        'min_impurity_decrease': [0.0, 0.01, 0.02]
    }

    model, metrics_df, results = models.random_forest_gridsearchcv_model(X_train, X_test, y_train, y_test, param_grid, 5, 'f1_macro')

    logging.info(f"criterio:{results['criterion'][0]}:max_depth:{results['max_depth'][0]}:min_impurity_decrease:{results['min_impurity_decrease'][0]}:min_samples_split:{results['min_samples_split'][0]}:n_estimators:{results['n_estimators'][0]}")
    logging.info(f"accuracy:{metrics_df['Valore'][0]}:precision:{metrics_df['Valore'][1]}:recall:{metrics_df['Valore'][2]}:f1_score:{metrics_df['Valore'][3]}:roc_auc:{metrics_df['Valore'][4]}")

