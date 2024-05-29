import pandas as pd

from imblearn.over_sampling import SMOTENC

from sklearn.preprocessing import StandardScaler

import logging

# creare un sottoinsieme del dataset originale con un rapporto 2:1 sulla feature LUX_01
def sub_dataset_cani(dataset):
    # divido il dataset sulla base dell'outcome
    dataset_0 = dataset[dataset['LUX_01']==0]
    dataset_1 = dataset[dataset['LUX_01']==1]

    # seleziono tutti i casi negativi in cui abbiamo almeno un valore nan 
    cols = ['ALO', 'HIPRL', 'STEMANTEVERSIONREAL', 'BCS']
    valori_nan = dataset_0.loc[dataset[cols].isna().any(axis=1)]

    # seleziono randomicamente altri casi negativi in modo da avere un rapporto 2:1
    dataset_0 = dataset_0.dropna(subset=cols, how='any') # elimino i casi negativi con almeno un valore nan per evitare che vengano selezionate 2 volte
    subset = dataset_0.sample(n=278, random_state=42) # seleziono randomicamente dal dataset originale con i soli casi negativi

    # lista dataframe da concatenare
    lista_dataframe = [dataset_1, valori_nan, subset]

    new_dataset = pd.concat(lista_dataframe, ignore_index=True)
    
    return new_dataset  # restituisco un dataset in cui ho un rapporto 2:1 tra le due classe di LUX_01


# metodo per la pulizia del dataset
def clean_dataset(dataset):

    dataset.AGEATSURGERYmo = dataset.AGEATSURGERYmo.str.replace(',', '.').astype('float64')
    dataset.BODYWEIGHTKG = dataset.BODYWEIGHTKG.str.replace(',', '.').astype('float64')
    dataset.BCS = dataset.BCS.str.replace(',', '.').astype('float64')
    dataset.STEMANTEVERSIONREAL = dataset.STEMANTEVERSIONREAL.str.replace(',', '.').astype('float64')

    dataset['HIPRL'] = dataset.HIPRL.replace('L', 0)
    dataset['HIPRL'] = dataset.HIPRL.replace('l', 0)
    dataset['HIPRL'] = dataset.HIPRL.replace('R', 1)

    dataset['RECTUSFEMORISM.RELEASE'] = dataset['RECTUSFEMORISM.RELEASE'].replace('NO', 0)
    dataset['RECTUSFEMORISM.RELEASE'] = dataset['RECTUSFEMORISM.RELEASE'].replace(' NO', 0)
    dataset['RECTUSFEMORISM.RELEASE'] = dataset['RECTUSFEMORISM.RELEASE'].replace('YES', 1)
    dataset['RECTUSFEMORISM.RELEASE'] = dataset['RECTUSFEMORISM.RELEASE'].replace('PARTIAL', 2)

    valori = dataset.STEMSIZE.unique()
    count = 0
    for v in valori:
        dataset['STEMSIZE'] = dataset['STEMSIZE'].replace(v, count)
        count += 1

    dataset['DIRECTION'] = dataset['DIRECTION'].replace('CRANIO-DORSALE', 0)
    dataset['DIRECTION'] = dataset['DIRECTION'].replace('CAUDO-VENTRALE', 1)

    return dataset


# metodo per eliminare le colonne per la predizione
def drop_cols(dataset):
    
    dataset = dataset.drop(['CASE_ID', 'n_luxation', 'first_lux_days_after_thr', 'DIRECTION'], axis=1)
    
    return dataset


# metodo per fare oversampling con SMOTENC
def oversampling(X, y):

    categorical_features = [True if dtype.name == 'int64'else False for dtype in X.dtypes]
    smotenc = SMOTENC(categorical_features=categorical_features, random_state=42)

    X_resampled, y_resampled = smotenc.fit_resample(X, y)

    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)  

    X_resampled['target'] = y_resampled

    return X_resampled


def scaler(dataset):
    
    scaler = StandardScaler()
    dataset[dataset.select_dtypes(include=['float64']).columns] = scaler.fit_transform(dataset.select_dtypes(include=['float64']))

    return dataset


def create_log(name):
    # Crea un oggetto logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Crea un handler per il file di log
    fh = logging.FileHandler(name)
    fh.setLevel(logging.INFO)

    # Crea un formatter per il messaggio di log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Aggiungi l'handler al logger
    logger.addHandler(fh)

    return logger