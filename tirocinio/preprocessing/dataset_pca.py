"""
    Questo script utilizza la PCA per ridurre la dimensionalità del dataset con one hot encoding.
    Effettua la standardizzazione dei dati, applica la PCA, e salva i risultati in due file CSV:
    uno contenente i dati trasformati e l'altro con i coefficienti dei componenti principali.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("../csv/dataset_dummy_feature.csv")

dataset['LUX_01'] = dataset['LUX_01'].astype(int)

X = dataset.drop(['LUX_01'], axis=1)
y = dataset['LUX_01']

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

features = X.columns

pca = PCA(n_components=85)
X_pca = pca.fit_transform(X_std)

loadings = pca.components_

# Creazione di un DataFrame per visualizzare i coefficienti
df_loadings = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(85)], index=features)

df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(85)])
df['LUX_01'] = y

df.to_csv("../csv/dataset_pca.csv", index=False)
df_loadings.to_csv("../csv/components_pca.csv", index=False)