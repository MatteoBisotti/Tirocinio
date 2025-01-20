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
    Carica il dataset dal file CSV, fa imputation dei valori mancanti con media e 
    mediana e lo pulisce con la funzione clean_dataset()

    Returns:
        pd.DataFrame: Il dataset pulito con imputation con media e mediana.
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

def train_test(dataset_original, dataset_oversampling, random):
    """
    Suddivide il dataset in set di addestramento e di test. Il test set è costituito da 300 elementi, 200 elementi presi dalla classe negativi
    mentre i restanti 100 presi dalla classe positiva. Tutti gli elementi che compongono il test set sono presi dal dataset originale, quindi 
    sono tutti casi reali, e non ottenuti con le tecniche di oversampling adottate. Nel segnatura del metodo vengono presi due dataset, il primo che 

    Args:
        dataset (pd.DataFrame): Il dataset originale.
        df (pd.DataFrame): Il dataset su cui effettuare la suddivisione.
        random (bool): Se True, la suddivisione è casuale; altrimenti, viene utilizzata una suddivisione fissa.

    Returns:
        il training set e il testing set
    """
    if random:
        y_test_0 = dataset_original[dataset_original['LUX_01']==0].sample(n=200)
        y_test_1 = dataset_original[dataset_original['LUX_01']==1].sample(n=100)
    else:
        y_test_0 = dataset_original[dataset_original['LUX_01']==0].sample(n=200, random_state=42)
        y_test_1 = dataset_original[dataset_original['LUX_01']==1].sample(n=100, random_state=42)

    
    lista_dataframe = [y_test_0, y_test_1]
    testing_set = pd.concat(lista_dataframe, ignore_index=True)

    merged_df = dataset_oversampling.merge(testing_set, on=dataset_oversampling.columns.tolist(), how='inner', indicator=True)
    righe_comuni = merged_df[merged_df['_merge'] == 'both'].drop(columns='_merge')

    merged_df = dataset_oversampling.merge(righe_comuni, on=dataset_oversampling.columns.tolist(), how='left', indicator=True)
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
    Calcola il numero di esempi positivi per l'oversampling.

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

def display_corr_matrix(dataset, title):
    '''
    Visualizza la matrice di correlazione del dataset tramite heatmap del modulo seaborn e la restituisce.

    Args:
        dataset (pd.Dataframe): Il dataset su cui calcolare la matrice di correlazione.

    Returns:
        matrice_corr: matrice di correlazione del dataset.
    '''
    matrice_corr = dataset.corr()

    plt.figure(figsize=(28, 20))

    sns.heatmap(matrice_corr, annot=True, 
                        fmt='.1f', cmap='coolwarm', 
                        square=True, linewidths=.5, 
                        cbar_kws={'shrink': .5},
                        vmin=-1.0, vmax=1.0)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(title, fontsize=25)

    return matrice_corr

def plot_boxplot(original, imputed, colonna):
    """
    Traccia i boxplot prima e dopo l'imputazione, evidenziando i valori imputati in rosso.
    
    Args:
        original (pd.DataFrame): Dataset con valori mancanti.
        imputed (pd.DataFrame): Dataset con valori imputati.
        colonna (str): Nome della colonna da visualizzare.
    """
    imputati_idx = original[colonna].isna()
    
    dati_originali = original[colonna].dropna() 
    dati_imputati = imputed[colonna] 
    valori_imputati = imputed.loc[imputati_idx, colonna]  

    fig, ax = plt.subplots(figsize=(10, 6))
    
    box_originali = ax.boxplot(
        [dati_originali],
        positions=[1],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="white", color="black"),
        medianprops=dict(color="black")
    )
    box_imputati = ax.boxplot(
        [dati_imputati],
        positions=[2],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="white", color="black"),
        medianprops=dict(color="black")
    )
    
    ax.scatter(
        x=[2] * len(valori_imputati), 
        y=valori_imputati,
        color="red",
        zorder=3,
        label="Valori imputati"
    )
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Prima dell'imputazione", "Dopo l'imputazione"])
    ax.set_title(f"Boxplot di '{colonna}' prima e dopo l'imputazione")
    ax.set_ylabel(colonna)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.show()