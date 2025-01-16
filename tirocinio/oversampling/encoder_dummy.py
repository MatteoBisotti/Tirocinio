from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def get_strategy_oversampling(n_negativi, rapporto):
    """
    Calcola il numero di esempi positivi per il sovracampionamento.

    Args:
        n_negativi (int): Il numero di esempi negativi.
        rapporto (float): Il rapporto desiderato tra esempi positivi e negativi.

    Returns:
        n_positivi: Il numero di esempi positivi da generare.
    """
    n_positivi = (1 / rapporto * n_negativi) - n_negativi
    return int(n_positivi)

def encoder(df, random_seed=42):
    """
    Applica il sovracampionamento tramite MLPRegressor per riequilibrare le classi nel dataset.

    Args:
        df (pd.DataFrame): Il dataset originale.
        oversampling_ratio (float): Il rapporto di sovracampionamento desiderato.
        random_seed (int): Il seme per la riproducibilità.

    Returns:
        pd.DataFrame: Il dataset riequilibrato.
    """
    positive_class = df[df['LUX_01'] == 1]
    negative_class = df[df['LUX_01'] == 0]

    X_pos = positive_class.drop(columns=['LUX_01'])
    y_pos = positive_class['LUX_01']

    scaler = StandardScaler()
    X_pos_scaled = scaler.fit_transform(X_pos)

    input_dim = X_pos_scaled.shape[1]
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=115, random_state=random_seed, solver='sgd', warm_start=True)
    mlp_regressor.fit(X_pos_scaled, X_pos_scaled)

    num_new_samples = get_strategy_oversampling(len(negative_class), 2/3) - len(positive_class)
    new_samples_input = np.random.rand(num_new_samples, input_dim)
    new_samples_scaled = mlp_regressor.predict(new_samples_input)
    new_samples = scaler.inverse_transform(new_samples_scaled)

    new_positive_class = pd.DataFrame(new_samples, columns=X_pos.columns)
    new_positive_class['LUX_01'] = 1

    y_test = new_positive_class.drop(['LUX_01'], axis=1).sample(n=141, random_state=42)
    mse = mean_squared_error(X_pos, y_test)

    dataset = pd.concat([df, new_positive_class], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    dataset['mse'] = mse

    return dataset