"""
Questo script carica il dataset originale, applica oversampling utilizzando l'algoritmo SMOTENC
e salva il dataset risultante in un file CSV.

Funzioni:
    - load_csv: Carica il dataset dal file CSV.
    - oversampling: Applica il sovracampionamento SMOTENC per bilanciare il dataset.

Moduli esterni richiesti:
    - pandas: Utilizzato per la manipolazione dei DataFrame.
    - sys: Utilizzato per aggiungere percorsi al path di ricerca del modulo.
    - functions (da base_lib): Funzioni di utilità per la gestione del dataset.
    - smotenc (da Oversampling): Modulo per l'applicazione del sovracampionamento SMOTENC.
"""

import pandas as pd
import sys

sys.path.append("../base_lib")
import functions as func

sys.path.append("../Oversampling")
import smotenc as smo

dataset = func.load_csv()

X = dataset.drop(['LUX_01'], axis=1)
y = dataset['LUX_01']

dataset_oversampling = smo.oversampling(dataset, X, y)

dataset_oversampling.to_csv("../csv/dataset_SMOTENC.csv", index=False)