"""
    Questo script esegue un'analisi della PCA (Principal Component Analysis) su un dataset,
    mostrando come varia la varianza spiegata cumulativa in funzione del numero di componenti principali.
    Il risultato è presentato sia in formato tabellare che grafico.
"""

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np

import sys
sys.path.append("../base_lib")
import models 
import functions as func

def main(dataset):
    display(Markdown("# Applicazione di PCA sul dataset"))
    display(dataset.head(5))

    X = dataset.drop(['LUX_01'], axis=1)
    y = dataset['LUX_01']

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X) 
    # Lista per memorizzare i risultati
    results = []

    # Eseguire la PCA per un numero di componenti variabile
    for n_components in range(5, 124, 5):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_std)
        
        # Varianza spiegata
        explained_variance = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance)
        
        # Aggiungi risultati alla lista
        results.append({
            'Numero componenti': n_components,
            'Varianza spiegata cumulata': cumulative_explained_variance[-1]
        })

    pca = PCA(n_components=124)
    X_pca = pca.fit_transform(X_std)
        
    # Varianza spiegata
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)
    
    # Aggiungi risultati alla lista
    results.append({
        'Numero componenti': 124,
        'Varianza spiegata cumulata': cumulative_explained_variance[-1]
    })


    # Creazione di un DataFrame per visualizzare i risultati
    df_results = pd.DataFrame(results)

    display(Markdown("### Risultati della PCA con la varianza spiegata cumulata e il numero di componenti"))

    # Stampa i risultati
    display(df_results)

    # Visualizzazione grafica dei risultati
    plt.figure(figsize=(12, 8))
    plt.plot(df_results['Numero componenti'], df_results['Varianza spiegata cumulata'], marker='o')
    plt.xlabel('Numero di componenti')
    plt.ylabel('Varianza spiegata cumulativa')
    plt.ylim(0, 1.02)
    plt.show()

    