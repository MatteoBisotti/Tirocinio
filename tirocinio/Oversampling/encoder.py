"""
Questo modulo fornisce funzioni per il sovracampionamento dei dataset, incluso il calcolo
del numero di esempi positivi necessari e l'applicazione del sovracampionamento tramite
un MLP Regressor.

Funzioni:
    - get_strategy_oversampling(n_negativi, rapporto): Calcola il numero di esempi positivi per il sovracampionamento.
    - encoder(df, binary_features): Applica il sovracampionamento tramite MLP Regressor per riequilibrare la feature LUX_01 nel dataset.

Moduli esterni richiesti:
    - pandas
    - sklearn.preprocessing.StandardScaler
    - tensorflow.keras.models.Sequential
    - tensorflow.keras.layers.Dense, Dropout
    - tensorflow.keras.optimizers.Adam
    - numpy
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

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

def encoder(df, binary_features):
    """
    Applica il sovracampionamento tramite MLP Regressor per riequilibrare la feature LUX_01 nel dataset.

    Args:
        df (pd.DataFrame): Il dataset originale.
        binary_features (pd.Series): Insieme delle feature con valore binario.

    Returns:
        pd.DataFrame: Il dataset bilanciato.
    """

    positive_class = df[df['LUX_01'] == 1]
    negative_class = df[df['LUX_01'] == 0]

    X_pos = positive_class.drop(columns=['LUX_01'])
    y_pos = positive_class['LUX_01']

    scaler = StandardScaler()
    X_pos_scaled = scaler.fit_transform(X_pos)

    input_dim = X_pos_scaled.shape[1]
    mlp_regressor = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(input_dim, activation='linear')
    ])

    mlp_regressor.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    mlp_regressor.fit(X_pos_scaled, X_pos_scaled, epochs=50, batch_size=32, validation_split=0.2)

    num_new_samples = get_strategy_oversampling(len(negative_class), 2/3) - len(positive_class)
    new_samples_scaled = mlp_regressor.predict(np.random.rand(num_new_samples, input_dim))
    new_samples = scaler.inverse_transform(new_samples_scaled)

    new_positive_class = pd.DataFrame(new_samples, columns=X_pos.columns)
    new_positive_class['LUX_01'] = 1

    for col in binary_features:
        if col in positive_class.columns:
            new_positive_class[col] = new_positive_class[col].apply(lambda x: round(x))

    dataset = pd.concat([df, new_positive_class], ignore_index=True)

    dataset = dataset.sample(frac=1).reset_index(drop=True)

    return dataset
