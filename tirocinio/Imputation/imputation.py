import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from imputers.mean_imputer import MeanImputer
from imputers.median_imputer import MedianImputer
from imputers.linear_regression_imputer import LinearRegressionImputer
from imputers.logistic_regression_imputer import LogisticRegressionImputer
from imputers.knn_distance_imputer import KnnDistanceImputer
from imputers.knn_uniform_imputer import KnnUniformImputer

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

    return dataset


# metodo per eliminare le colonne per la predizione
def drop_cols(dataset):
    
    dataset = dataset.drop(['CASE_ID', 'n_luxation', 'first_lux_days_after_thr', 'DIRECTION'], axis=1)
    
    return dataset


# metodo per fare imputation con media
def imputation_mean(dataset, name_feature):

    mean = MeanImputer(dataset=dataset, name_feature=name_feature)
    dataset = mean.impute()

    return dataset


# metodo per fare imputation con mediana
def imputation_median(dataset, name_feature):

    median = MedianImputer(dataset=dataset, name_feature=name_feature)
    dataset = median.impute()

    return dataset


# metodo per imputare l'intero dataset con media/mediana
def total_imputation_mean_median(dataset):

    dataset = imputation_mean(dataset, 'BCS')
    dataset = imputation_median(dataset, 'STEMANTEVERSIONREAL')
    dataset = imputation_mean(dataset, 'ALO')
    dataset = imputation_median(dataset, 'HIPRL')

    return dataset


# metodo per fare imputation con regressione lineare
def imputation_linear_regression(dataset, name_feature):

    model = LinearRegressionImputer(dataset=dataset, name_feature=name_feature)
    dataset = model.impute_value()

    return dataset


# metodo per fare imputation con regressione logistica
def imputation_logistic_regression(dataset, name_feature):

    model = LogisticRegressionImputer(dataset=dataset, name_feature=name_feature)
    dataset = model.impute_value()

    return dataset


# metodo per imputare l'intero dataset con regressione
def total_imputation_regression(dataset):

    dataset = imputation_linear_regression(dataset, 'BCS')
    dataset = imputation_linear_regression(dataset, 'STEMANTEVERSIONREAL')
    dataset = imputation_linear_regression(dataset, 'ALO')
    dataset = imputation_logistic_regression(dataset, 'HIPRL')

    return dataset


# metoto per imputare l'intero dataset con Knn imputation (weights='uniform')
def knn_imputation_uniform(dataset, n_neighbors):

    knn_imputer = KnnUniformImputer(n_neighbors=n_neighbors)
    dataset = knn_imputer.impute_value(dataset=dataset)

    # Identifichiamo le colonne che devono essere mantenute come int
    int_columns = [
        'CASE_ID', 'BREED', 'GENDER_01', 'Taglia', 'INDICATIONFORTHR',
        'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE',
        'CUPRETROVERSION', 'RECTUSFEMORISM.RELEASE', 'LUX_01', 'LUX_CR'
    ]

    # Converte le colonne specifiche di nuovo al tipo originale, se necessario
    for col in int_columns:
        dataset[col] = dataset[col].round().astype('int64')

    return dataset


# metodo per imputare l'intero dataset con Knn imputation (weights='distance')
def knn_imputation_distance(dataset, n_neighbors):

    knn_imputer = KnnDistanceImputer(n_neighbors=n_neighbors)
    dataset = knn_imputer.impute_value(dataset=dataset)

    # Identifichiamo le colonne che devono essere mantenute come int
    int_columns = [
        'CASE_ID', 'BREED', 'GENDER_01', 'Taglia', 'INDICATIONFORTHR',
        'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE',
        'CUPRETROVERSION', 'RECTUSFEMORISM.RELEASE', 'LUX_01', 'LUX_CR'
    ]

    # Converte le colonne specifiche di nuovo al tipo originale, se necessario
    for col in int_columns:
        dataset[col] = dataset[col].round().astype('int64')

    return dataset