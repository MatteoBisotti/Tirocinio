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
def knn_imputation(dataset, binary_features):

    knn_imputer = ImputerKNN()
    dataset = knn_imputer.impute_value(dataset=dataset)

    # per ogni variabile binaria assegna 1 se il valore è maggiore di 0.5 oppure 0 altrmenti
    for col in binary_features:
        dataset[col] = dataset[col].apply(lambda x: round(x) if not (x == 0 or x == 1) else x)

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


# metodo per imputation con rete neurale con funzione di attivazione 'linear' (variabili continue)
def nr_imputation_linear(dataset, name_feature):

    imputer = LinearNrImputer(dataset)
    dataset[name_feature] = imputer.impute_missing_value(dataset, name_feature)
    
    return dataset[name_feature]


# metodo per imputation con rete neurale con funzione di attivazione 'sigmoid' (variabili binarie)
def nr_imputation_sigmoid(dataset, name_feature):

    imputer = SoftmaxNrImputer(dataset)
    dataset[name_feature] = imputer.impute_missing_value(dataset, name_feature)
    
    return dataset[name_feature]


# metodo per imputare l'intero dataset con rete neurale, con imputazione una feature per volta 
def total_nr_imputation(dataset):

    dataset['ALO'] = nr_imputation_linear(dataset, 'ALO')
    dataset['STEMANTEVERSIONREAL'] = nr_imputation_linear(dataset, 'STEMANTEVERSIONREAL')
    dataset['BSC'] = nr_imputation_linear(dataset, 'BCS')
    dataset['HIPRL'] = nr_imputation_sigmoid(dataset, 'HIPRL')

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


# stampa grafici per visualizzare i nuovi valori generati
def print_boxplot(dataset, features):

    plt.figure(figsize=(14, 16))