import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Input, Dense, Dropout
from keras.models import Model

import sys 
sys.path.append("../base_lib")
import functions as func

# Funzione Mixup per la generazione dei nuovi campioni
def mixup(x1, x2, alpha=0.2):
    lambda_ = np.random.beta(alpha, alpha)
    return lambda_ * x1 + (1 - lambda_) * x2

dataset = pd.read_csv("../csv/dataset_original.csv")
dataset = func.drop_cols(dataset)

# Divisione del dataset tra positivi e negativi
X = dataset.drop(['LUX_01'], axis=1)
y = dataset['LUX_01']

X_pos = X[y == 1]
scaler = StandardScaler()
X_pos_scaled = scaler.fit_transform(X_pos)

# Definizione dell'autoencoder
input_dim = X_pos_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Addestramento dell'autoencoder
autoencoder.fit(X_pos_scaled, X_pos_scaled, epochs=100, batch_size=32, shuffle=True)

n_samples_to_generate = 1387
generated_samples = []

for _ in range(n_samples_to_generate):
    idx1, idx2 = np.random.choice(len(X_pos_scaled), 2, replace=False)
    mixed_sample = mixup(X_pos_scaled[idx1], X_pos_scaled[idx2])
    generated_samples.append(mixed_sample)

generated_samples = np.vstack(generated_samples)
generated_samples_original_scale = scaler.inverse_transform(generated_samples)

new_samples_df = pd.DataFrame(generated_samples_original_scale, columns=X.columns)
new_samples_df['LUX_01'] = 1

dataset_augmented = pd.concat([dataset, new_samples_df], ignore_index=True)

# Calcola l'MSE tra i campioni originali e i campioni generati
X_pos_reconstructed = autoencoder.predict(X_pos_scaled)
np.random.seed(42)
sample_indices = np.random.choice(generated_samples.shape[0], size=141, replace=False)

sample = generated_samples[sample_indices]

mse_generated = mean_squared_error(X_pos_scaled, sample)

dataset_augmented['mse'] = mse_generated

dataset_augmented.to_csv("../csv/dataset_original_autoencoder.csv", index=False)