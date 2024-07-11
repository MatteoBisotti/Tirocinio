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
    Applica il sovracampionamento tramite algoritmo SMOTENC per riequilibrare le classi nel dataset.

    Args:
        dataset (pd.DataFrame): Il dataset originale.
        X (pd.DataFrame): Le caratteristiche del dataset.
        y (pd.Series): Le etichette del dataset.

    Returns:
        pd.DataFrame: Il dataset riequilibrato con sovracampionamento.
    """
    categorical_features = [dtype.name == 'int64' for dtype in X.dtypes]

    n_negative = len(dataset[dataset['LUX_01']==0])
    n_positive = get_strategy_oversampling(n_negative, 2/3)

    sampling_strategy = {0: n_negative, 1: n_positive}
    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42, sampling_strategy=sampling_strategy)

    X_resampled, y_resampled = smote_nc.fit_resample(X, y)

    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled_df = pd.Series(y_resampled, name='LUX_01')

    dataset = pd.concat([X_resampled_df, y_resampled_df], axis=1)
    dataset = dataset.sample(random_state=42, frac=1)

    return dataset