import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

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


# metodo per fare imputation con media aritmetica
def imputation_mean(dataset, name_feature):
    mean = dataset[name_feature].mean()

    dataset[name_feature] = dataset[name_feature].fillna(mean)

    return dataset


# metodo per fare imputation con mediana
def imputation_median(dataset, name_feature):
    median = dataset[name_feature].median()

    dataset[name_feature] = dataset[name_feature].fillna(median)

    return dataset


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


# metodo per imputare il dataset con media/mediana
def total_imputation_mean_median(dataset):

    dataset = imputation_mean(dataset, 'BCS')
    dataset = imputation_median(dataset, 'STEMANTEVERSIONREAL')
    dataset = imputation_mean(dataset, 'ALO')
    dataset = imputation_median(dataset, 'HIPRL')

    return dataset


# metodo per fare imputation con regressione lineare
def imputation_linear_regression(dataset, name_feature):

    # divido il dataset tra i casi che non hanno la feature nan
    dataset_notnull = dataset.dropna()
    # e i casi che hanno quella feature nan
    dataset_null = dataset[dataset[name_feature].isnull()]

    X_train = dataset_notnull.drop([name_feature], axis=1)
    y_train = dataset_notnull[name_feature]

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_test = dataset_null.drop([name_feature], axis=1)
    predict_values = model.predict(X_test)

    dataset.loc[dataset[name_feature].isnull(), name_feature] = predict_values

    return dataset


# metodo per fare imputation con regressione logistica
def imputation_logistic_regression(dataset, name_feature):

    dataset_notnull = dataset.dropna()
    dataset_null = dataset[dataset[name_feature].isnull()]

    X_train = dataset_notnull.drop([name_feature], axis=1)
    y_train = dataset_notnull[name_feature]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    X_test = dataset_null.drop([name_feature], axis=1)
    predict_values = model.predict(X_test)

    dataset.loc[dataset[name_feature].isnull(), name_feature] = predict_values

    return dataset


# metodo per imputare il dataset con regressione
def total_imputation_regression(dataset):

    dataset = imputation_linear_regression(dataset, 'BCS')
    dataset = imputation_linear_regression(dataset, 'STEMANTEVERSIONREAL')
    dataset = imputation_linear_regression(dataset, 'ALO')
    dataset = imputation_logistic_regression(dataset, 'HIPRL')

    return dataset