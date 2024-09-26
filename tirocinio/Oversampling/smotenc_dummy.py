"""
Questo modulo fornisce funzioni per il sovracampionamento dei dataset, incluso il calcolo
del numero di esempi positivi necessari e l'applicazione del sovracampionamento tramite
algoritmo SMOTENC.

Funzioni:
    - get_strategy_oversampling(n_negativi, rapporto): Calcola il numero di esempi positivi per il sovracampionamento.
    - oversampling(dataset, X, y): Applica il sovracampionamento tramite algoritmo SMOTENC per bilanciare la feature LUX_01 nel dataset.

Moduli esterni richiesti:
    - pandas
    - imblearn.over_sampling.SMOTENC
"""

import pandas as pd
from imblearn.over_sampling import SMOTENC

def get_strategy_oversampling(n_negativi, rapporto):
    """
    Calcola il numero di esempi positivi per il sovracampionamento.

    Args:
        n_negativi (int): Il numero di esempi negativi.
        rapporto (float): Il rapporto desiderato tra esempi positivi e negativi.

    Returns:
        int: Il numero di esempi positivi da generare.
    """
    n_positivi = (1 / rapporto * n_negativi) - n_negativi
    return int(n_positivi)

def oversampling(dataset, X, y):
    """
    Applica il sovracampionamento tramite algoritmo SMOTENC per bilanciare la feature LUX_01 nel dataset.

    Args:
        dataset (pd.DataFrame): Il dataset originale.
        X (pd.DataFrame): Feature del dataset.
        y (pd.Series): Feature LUX_01.

    Returns:
        pd.DataFrame: Il dataset bilanciato.
    """

    n_negative = len(dataset[dataset['LUX_01']==0])
    n_positive = get_strategy_oversampling(n_negative, 2/3)

    sampling_strategy = {0: n_negative, 1: n_positive}

    # Seleziona le colonne che hanno il tipo di dato int64
    cat_columns = dataset.select_dtypes(include=['int64']).columns

    # Ottieni gli indici delle colonne int64
    cat_col_index = [dataset.columns.get_loc(col) for col in cat_columns]

    sm = SMOTENC(categorical_features=cat_col_index, random_state=123, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled_df = pd.Series(y_resampled, name='LUX_01')

    df = pd.concat([X_resampled_df, y_resampled_df], axis=1)
    df = df.sample(random_state=42, frac=1)

    return dataset
