"""
Questo modulo contiene funzioni per la manipolazione, pulizia, suddivisione e riequilibrio di un dataset specifico,
oltre a funzioni per il logging e la visualizzazione dei dati.

Funzioni:
    sub_dataset_cani(dataset): Crea un nuovo dataset basato su un subset di dati di cani.
    clean_dataset(dataset): Pulisce il dataset applicando varie trasformazioni ai dati.
    load_csv(): Carica il dataset dal file CSV e lo pulisce.
    drop_cols(dataset): Rimuove colonne specifiche dal dataset.
    train_test(dataset, df, random): Suddivide il dataset in set di addestramento e di test.
    oversampling(dataset, X, y): Applica il sovracampionamento tramite algoritmo SMOTENC per riequilibrare le classi nel dataset.
    oversampling_SMOTE(dataset, X, y): Applica il sovracampionamento tramite algoritmo SMOTE per riequilibrare le classi nel dataset.
    encoder(dataset, binary_features): Applica il sovracampionamento tramite MLPregressor per riequilibrare le classi nel dataset.
    scaler(dataset): Applica la standardizzazione alle caratteristiche numeriche del dataset.
    get_strategy_oversampling(n_negativi, rapporto): Calcola il numero di esempi positivi per il sovracampionamento.
    plot_outcome_feature(df, feature_name): Visualizza la distribuzione dei valori di una caratteristica specifica.
    display_corr_matrix(dataset): Visualizza la matrice di correlazione del dataset tramite heatmap del modulo seaborn e la restituisce.

Moduli esterni richiesti:
    pandas
    imblearn.over_sampling.SMOTENC
    imblearn.over_sampling.SMOTE
    sklearn.preprocessing.StandardScaler
    matplotlib.pyplot
    seaborn
    sys
    imp (modulo per l'imputazione dei dati)
    sklearn.preprocessing
    tensorflow.keras.models
    tensorflow.keras.layers
    tensorflow.keras.optimizers
    numpy
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../Imputation")
import imputation as imp

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def sub_dataset_cani(dataset):
    """
    Crea un nuovo dataset basato su un subset di dati di cani.

    Args:
        dataset (pd.DataFrame): Il dataset originale.

    Returns:
        pd.DataFrame: Il nuovo dataset con un subset di dati.
    """
    dataset_0 = dataset[dataset['LUX_01']==0]
    dataset_1 = dataset[dataset['LUX_01']==1]

    cols = ['ALO', 'HIPRL', 'STEMANTEVERSIONREAL', 'BCS']
    valori_nan = dataset_0.loc[dataset[cols].isna().any(axis=1)]

    dataset_0 = dataset_0.dropna(subset=cols, how='any') 
    subset = dataset_0.sample(n=278, random_state=42) 

    lista_dataframe = [dataset_1, valori_nan, subset]

    new_dataset = pd.concat(lista_dataframe, ignore_index=True)
    
    return new_dataset  

def clean_dataset(dataset):
    """
    Pulisce il dataset applicando varie trasformazioni ai dati.

    Args:
        dataset (pd.DataFrame): Il dataset da pulire.

    Returns:
        pd.DataFrame: Il dataset pulito.
    """
    dataset = pd.read_csv("../data/datiLussazioniDefinitivi.csv", delimiter=";")

    dataset.AGEATSURGERYmo = dataset.AGEATSURGERYmo.str.replace(',', '.').astype('float64')
    dataset.BODYWEIGHTKG = dataset.BODYWEIGHTKG.str.replace(',', '.').astype('float64')
    dataset.BCS = dataset.BCS.str.replace(',', '.').astype('float64')
    dataset.STEMANTEVERSIONREAL = dataset.STEMANTEVERSIONREAL.str.replace(',', '.').astype('float64')

    dataset['HIPRL'] = dataset.HIPRL.replace(['L', 'l'], 0)
    dataset['HIPRL'] = dataset.HIPRL.replace('R', 1)

    dataset['RECTUSFEMORISM.RELEASE'] = dataset['RECTUSFEMORISM.RELEASE'].replace(['NO', ' NO'], 0)
    dataset['RECTUSFEMORISM.RELEASE'] = dataset['RECTUSFEMORISM.RELEASE'].replace('YES', 1)
    dataset['RECTUSFEMORISM.RELEASE'] = dataset['RECTUSFEMORISM.RELEASE'].replace('PARTIAL', 2)

    valori = dataset.STEMSIZE.unique()
    count = 0
    for v in valori:
        dataset['STEMSIZE'] = dataset['STEMSIZE'].replace(v, count)
        count += 1

    dataset['DIRECTION'] = dataset['DIRECTION'].replace('CRANIO-DORSALE', 0)
    dataset['DIRECTION'] = dataset['DIRECTION'].replace('CAUDO-VENTRALE', 1)

    dataset = dataset.drop(['CASE_ID', 'HIPRL', 'INDICATIONFORTHR'], axis=1)
    cols = ['n_luxation', 'first_lux_days_after_thr', 'DIRECTION']
    dataset[cols] = dataset[cols].fillna(-1)

    return dataset

def load_csv():
    """
    Carica il dataset dal file CSV e lo pulisce.

    Returns:
        pd.DataFrame: Il dataset caricato e pulito.
    """
    dataset = pd.read_csv("../data/datiLussazioniDefinitivi.csv", delimiter=";")
    dataset = clean_dataset(dataset)
    dataset = imp.total_imputation_mean_median(dataset)
    return dataset

def drop_cols(dataset):
    """
    Rimuove colonne specifiche dal dataset.

    Args:
        dataset (pd.DataFrame): Il dataset originale.

    Returns:
        pd.DataFrame: Il dataset con le colonne rimosse.
    """
    dataset = dataset.drop(['n_luxation', 'first_lux_days_after_thr', 'DIRECTION'], axis=1)
    return dataset

def train_test(dataset, df, random):
    """
    Suddivide il dataset in set di addestramento e di test.

    Args:
        dataset (pd.DataFrame): Il dataset originale.
        df (pd.DataFrame): Il dataset su cui effettuare la suddivisione.
        random (bool): Se True, la suddivisione è casuale; altrimenti, viene utilizzato un seme fisso.

    Returns:
        tuple: Un tuple contenente il set di addestramento e il set di test.
    """
    if random:
        y_test_0 = dataset[dataset['LUX_01']==0].sample(n=200)
        y_test_1 = dataset[dataset['LUX_01']==1].sample(n=100)
    else:
        y_test_0 = dataset[dataset['LUX_01']==0].sample(n=200, random_state=42)
        y_test_1 = dataset[dataset['LUX_01']==1].sample(n=100, random_state=42)

    
    lista_dataframe = [y_test_0, y_test_1]
    testing_set = pd.concat(lista_dataframe, ignore_index=True)

    merged_df = df.merge(testing_set, on=df.columns.tolist(), how='inner', indicator=True)
    righe_comuni = merged_df[merged_df['_merge'] == 'both'].drop(columns='_merge')

    merged_df = df.merge(righe_comuni, on=df.columns.tolist(), how='left', indicator=True)
    training_set = merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')

    return training_set, testing_set

def oversampling(dataset, X, y):
    """
    Applica il sovracampionamento tramite algoritmo SMOTENC per riequilibrare le classi nel dataset.

    Args:
        dataset (pd.DataFrame): Il dataset originale.
        X (pd.DataFrame): Le caratteristiche del dataset.
        y (pd.Series): Le etichette del dataset.

    Returns:
        pd.DataFrame: Il dataset riequilibrato con sovracampionamento.
    """
    categorical_features = [dtype.name == 'int64' for dtype in X.dtypes]

    n_negative = len(dataset[dataset['LUX_01']==0])
    n_positive = get_strategy_oversampling(n_negative, 2/3)

    sampling_strategy = {0: n_negative, 1: n_positive}
    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42, sampling_strategy=sampling_strategy)

    X_resampled, y_resampled = smote_nc.fit_resample(X, y)

    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled_df = pd.Series(y_resampled, name='LUX_01')

    dataset = pd.concat([X_resampled_df, y_resampled_df], axis=1)
    dataset = dataset.sample(random_state=42, frac=1)

    return dataset

def oversampling_SMOTE(dataset, X, y):
    """
    Applica il sovracampionamento tramite algoritmo SMOTE per riequilibrare le classi nel dataset.

    Args:
        dataset (pd.DataFrame): Il dataset originale.
        X (pd.DataFrame): Le caratteristiche del dataset.
        y (pd.Series): Le etichette del dataset.

    Returns:
        pd.DataFrame: Il dataset riequilibrato con sovracampionamento.
    """
    n_negative = len(dataset[dataset['LUX_01']==0])
    n_positive = get_strategy_oversampling(n_negative, 2/3)

    sampling_strategy = {0: n_negative, 1: n_positive}

    smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled_df = pd.Series(y_resampled, name='LUX_01')

    df = pd.concat([X_resampled_df, y_resampled_df], axis=1)
    df = df.sample(random_state=42, frac=1)
    return df

def scaler(dataset):
    """
    Applica la standardizzazione alle caratteristiche numeriche del dataset.

    Args:
        dataset (pd.DataFrame): Il dataset originale.

    Returns:
        pd.DataFrame: Il dataset standardizzato.
    """
    scaler = StandardScaler()
    dataset[dataset.select_dtypes(include=['float64']).columns] = scaler.fit_transform(dataset.select_dtypes(include=['float64']))

    return dataset

def get_strategy_oversampling(n_negativi, rapporto):
    """
    Calcola il numero di esempi positivi per il sovracampionamento.

    Args:
        n_negativi (int): Il numero di esempi negativi.
        rapporto (float): Il rapporto desiderato tra esempi positivi e negativi.

    Returns:
        n_positivi: Il numero di esempi positivi da generare.
    """
    n_positivi = (1 / rapporto * n_negativi) - n_negativi
    return int(n_positivi)

def plot_outcome_feature(df, feature_name):
    """
    Visualizza la distribuzione dei valori di una caratteristica specifica.

    Args:
        df (pd.DataFrame): Il dataset contenente la caratteristica.
        feature_name (str): Il nome della caratteristica da visualizzare.
    """
    value_counts = df[feature_name].value_counts()

    value_counts_df = value_counts.reset_index()
    value_counts_df.columns = ['Value', 'Count']

    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x='Value', y='Count', data=value_counts_df, palette='deep')

    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.0f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

    plt.xlabel('Classe')
    plt.ylabel('Conteggio valori')
    plt.title('Distribuzione dei valori di {}'.format(feature_name))
    plt.show()

def display_corr_matrix(dataset):
    '''
    Visualizza la matrice di correlazione del dataset tramite heatmap del modulo seaborn e la restituisce.

    Args:
        dataset (pd.Dataframe): Il dataset su cui calcolare la matrice di correlazione.

    Returns:
        matrice_corr: matrice di correlazione del dataset.
    '''
    matrice_corr = dataset.corr()

    # Dimensione della figura
    plt.figure(figsize=(28, 20))

    # Informazioni della heatmap
    sns.heatmap(matrice_corr, annot=True, 
                        fmt='.1f', cmap='coolwarm', 
                        square=True, linewidths=.5, 
                        cbar_kws={'shrink': .5},
                        vmin=-1.0, vmax=1.0)

    # Descrizione delle etichette della matrice
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    return matrice_corr