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
    categorical_features = [dtype.name == 'int64' for dtype in X.dtypes]

    n_negative = len(dataset[dataset['LUX_01'] == 0])
    n_positive = get_strategy_oversampling(n_negative, 2/3)

    sampling_strategy = {0: n_negative, 1: n_positive}
    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=123, sampling_strategy=sampling_strategy)

    X_resampled, y_resampled = smote_nc.fit_resample(X, y)

    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled_df = pd.Series(y_resampled, name='LUX_01')

    dataset = pd.concat([X_resampled_df, y_resampled_df], axis=1)
    dataset = dataset.sample(random_state=42, frac=1).reset_index(drop=True)

    return dataset