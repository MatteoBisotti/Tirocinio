"""
Questo script carica un dataset utilizzando una funzione definita in un modulo esterno
e salva il dataset risultante in un file CSV.

Funzioni:
    - load_csv: Carica il dataset dal file CSV.

Moduli esterni richiesti:
    - pandas: Utilizzato per la manipolazione dei DataFrame.
    - sys: Utilizzato per aggiungere percorsi al path di ricerca del modulo.
    - functions (da base_lib): Funzioni di utilità per la gestione del dataset.
"""

import pandas as pd
import sys

sys.path.append("../base_lib")

import functions as func

dataset = func.load_csv()

dataset.to_csv("../csv/dataset_original.csv", index=False)