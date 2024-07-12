"""
Questo script carica un dataset, esegue un'operazione di encoding per il riequilibrio delle classi,
e salva il dataset risultante in un file CSV.

Funzioni:
    - load_csv: Carica il dataset dal file CSV.
    - encoder: Applica un'operazione per riequilibrare la feature LUX_01 nel dataset.

Moduli esterni richiesti:
    - pandas: Utilizzato per la manipolazione dei dataframe.
    - sys: Utilizzato per aggiungere percorsi al path di ricerca del modulo.
    - functions (da base_lib): Funzioni di utilità per la gestione del dataset.
    - encoder (da Oversampling): Funzioni bilanciare il dataset.
"""

import pandas as pd
import sys

sys.path.append("../base_lib")
import functions as func

sys.path.append("../Oversampling")
import encoder as enc

dataset = func.load_csv()

categorical_features = [dtype.name == 'int64' for dtype in dataset.dtypes]

augmented_df = enc.encoder(dataset, categorical_features)

augmented_df.to_csv("../csv/dataset_encoder.csv", index=False)