"""
Questo script carica il dataset originale, applica una one hot encoding
e salva il dataset risultante in un file CSV.

Moduli esterni richiesti:
    - pandas: per la manipolazione e analisi dei dati.
    - sys: per la gestione del percorso della libreria personalizzata.
    - functions (libreria personalizzata): contiene la funzione `load_csv()` per il caricamento del dataset.
"""

import pandas as pd
import sys

sys.path.append("../base_lib")

import functions as func

dataset = func.load_csv()

nonum_feats_names = ['BREED', 'GENDER_01', 'Taglia', 'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE', 'CUPRETROVERSION', 'RECTUSFEMORISM.RELEASE', 'LUX_01', 'LUX_CR']
num_cols_names = ['AGEATSURGERYmo', 'BODYWEIGHTKG', 'BCS', 'ALO', 'STEMANTEVERSIONREAL']

nonum_feats = dataset[nonum_feats_names].astype('category')
ohc_feats = pd.get_dummies(nonum_feats,drop_first=True)
ohc_feats.rename(columns={'LUX_01_1': 'LUX_01'}, inplace=True)

target = ohc_feats['LUX_01']
ohc_feats = ohc_feats.drop(['LUX_01'], axis=1)

dataset_dummy = pd.concat([dataset[num_cols_names], ohc_feats, target],axis=1)

dataset_dummy.to_csv("../csv/dataset_dummy_feature.csv", index=False)