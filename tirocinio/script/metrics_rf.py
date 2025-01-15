"""
    Questo script legge le metriche dei modelli di random forest da file di log, le organizza in un formato tabellare 
    e crea un grafico a barre raggruppato per confrontare diverse metriche (Accuracy, Precision, Recall, F1-Score, ROC AUC e Specificity) 
    tra i vari esperimenti.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def append_metrics(path, acc, pre, rec, f1, roc, spe):
    with open(path, 'r') as file:
        righe = file.readlines()

    if len(righe) > 1:  # Controlliamo che ci sia almeno una riga di dati
        riga = righe[1].strip()
        metrics = riga.split(':')
        acc.append(metrics[1].strip())
        pre.append(metrics[3].strip())
        rec.append(metrics[5].strip())
        f1.append(metrics[7].strip())
        roc.append(metrics[9].strip())
        spe.append(metrics[11].strip())

def convert(lista):
    return [float(valore) for valore in lista]

def main():
    acc = []
    pre = []
    rec = []
    f1 = []
    roc = []
    spe = []

    # Percorsi dei file di log
    paths = [
        '../logs/rf_original_data_smotenc.log',
        '../logs/rf_original_data_autoencoder.log',
        '../logs/rf_dummy_data_smotenc.log',
        '../logs/rf_dummy_data_autoencoder.log',
        '../logs/rf_pca_data_smotenc.log',
        '../logs/rf_pca_data_autoencoder.log'
    ]

    # Aggiunta delle metriche per ogni file
    for path in paths:
        append_metrics(path, acc, pre, rec, f1, roc, spe)

    # Conversione delle liste in float
    acc = convert(acc)
    pre = convert(pre)
    rec = convert(rec)
    f1 = convert(f1)
    roc = convert(roc)
    spe = convert(spe)

    # Nomi dei modelli
    modelli = [
        'Original\nwith SMOTENC', 'Original\nwith Autoencoder', 
        'Dummy\nwith SMOTENC', 'Dummy\nwith Autoencoder',
        'PCA\nwith SMOTENC', 'PCA\nwith Autoencoder'
    ]

    # Creazione del DataFrame per Seaborn
    df = pd.DataFrame({
        'Modelli': modelli * 6,
        'Metriche': ['Accuracy'] * len(modelli) + ['Precision'] * len(modelli) + 
                    ['Recall'] * len(modelli) + ['F1'] * len(modelli) + 
                    ['ROC AUC'] * len(modelli) + ['Specificity'] * len(modelli),
        'Valori': acc + pre + rec + f1 + roc + spe
    })

    # Creazione del grafico a barre raggruppato con Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Modelli', y='Valori', hue='Metriche', data=df)
    plt.ylim(0, 1)

    # Impostare la legenda al centro
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()
