import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

class ImputerAllNr:

    def __init__(self, dataset):
        self.dataset = dataset

    def build_model(self, input_dim):
        model = Sequential([
            Dense(10, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(5, activation='relu'),
            Dropout(0.2),
            Dense(input_dim, activation='linear')  # Ricostruiamo tutti i valori
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    # Funzione per l'imputazione dei valori mancanti
    def impute_missing_values(self, model, scaler, binary_features):

        # identifico le feature continue nel dataset
        continuous_features = [col for col in self.dataset.columns if col not in binary_features]
        # insieme di booleani che identifica i valori nan nel dataset
        bool_nan = np.isnan(self.dataset.values)
        # faccio una copia del dataset
        data_scaled = self.dataset.copy()
        # le caratteristiche continue vengono standardizzate tramite scaler e i valori mancanti vengono riempiti momentaneamente con media 
        data_scaled[continuous_features] = scaler.transform(self.dataset[continuous_features].fillna(self.dataset[continuous_features].mean()))
        
        # si utilizza il modello di rete neurale per predire i valori mancanti 
        data_imputed_scaled = model.predict(data_scaled)
        data_imputed = self.dataset.copy()

        # le caratteristiche continue nel dataset predetto vengono riportate alla loro scala originale
        data_imputed[continuous_features] = scaler.inverse_transform(data_imputed_scaled[:, :len(continuous_features)])
        
        # per ogni variabile binaria assegna 1 se il valore è maggiore di 0.5 oppure 0 altrmenti
        for i, feature in enumerate(binary_features):
            data_imputed[feature] = np.where(data_imputed_scaled[:, len(continuous_features) + i] > 0.5, 1, 0)
        
        # sostituisce i valori mancanti con i valori imputati
        data_filled = self.dataset.copy()
        data_filled[bool_nan] = data_imputed[bool_nan]
        
        return data_filled 
    