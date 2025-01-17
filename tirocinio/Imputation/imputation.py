from imputers.mean_imputer import MeanImputer
from imputers.median_imputer import MedianImputer

from imputers.linear_regression_imputer import LinearRegressionImputer
from imputers.logistic_regression_imputer import LogisticRegressionImputer

from imputers.imputer_knn import ImputerKNN

from imputers.nr_one_linear_imputation import LinearNrImputer
from imputers.nr_one_softmax_imputation import SoftmaxNrImputer

from sklearn.preprocessing import StandardScaler

from imputers.nr_all_imputation import ImputerAllNr

import matplotlib.pyplot as plt


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
    dataset_imputed = dataset.copy()

    dataset_imputed = imputation_mean(dataset_imputed, 'BCS')
    dataset_imputed = imputation_median(dataset_imputed, 'STEMANTEVERSIONREAL')
    dataset_imputed = imputation_mean(dataset_imputed, 'ALO')

    return dataset_imputed


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
    dataset_imputed = dataset.copy()

    dataset_imputed = imputation_linear_regression(dataset_imputed, 'BCS')
    dataset_imputed = imputation_linear_regression(dataset_imputed, 'STEMANTEVERSIONREAL')
    dataset_imputed = imputation_linear_regression(dataset_imputed, 'ALO')

    return dataset_imputed


# metoto per imputare l'intero dataset con Knn imputation
def knn_imputation(dataset, binary_features):

    knn_imputer = ImputerKNN()
    dataset = knn_imputer.impute_value(dataset=dataset)

    # per ogni variabile binaria assegna 1 se il valore è maggiore di 0.5 oppure 0 altrmenti
    for col in binary_features:
        dataset[col] = dataset[col].apply(lambda x: round(x) if not (x == 0 or x == 1) else x)

    # Identifichiamo le colonne che devono essere mantenute come int
    int_columns = [
        'BREED', 'GENDER_01', 'Taglia',
        'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE',
        'CUPRETROVERSION', 'RECTUSFEMORISM.RELEASE', 'LUX_01', 'LUX_CR'
    ]

    # Converte le colonne specifiche di nuovo al tipo originale, se necessario
    for col in int_columns:
        dataset[col] = dataset[col].round().astype('int64')

    return dataset


# metodo per imputare l'intero dataset con rete neurale, con imputazione tutte le feature insieme
def nr_all_imputation(dataset, binary_features):

    # Separiamo le feature binarie dalle altre
    continuous_features = [col for col in dataset.columns if col not in binary_features]

    # Preprocessamento: Normalizziamo solo le feature continue
    scaler = StandardScaler()
    dataset_scaled = dataset.copy()
    dataset_scaled[continuous_features] = scaler.fit_transform(dataset[continuous_features].fillna(dataset[continuous_features].mean()))

    # Prepariamo i dati per l'addestramento del modello
    X_train = dataset_scaled.copy()

    imputer = ImputerAllNr(dataset)

    # Addestramento del modello
    model = imputer.build_model(X_train.shape[1])
    model.fit(X_train, X_train, epochs=50, validation_split=0.2, verbose=0)

    data_filled = imputer.impute_missing_values(model, scaler, binary_features)

    return data_filled