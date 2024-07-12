"""
Questo modulo contiene funzioni per l'addestramento di un modello Random Forest Classifier con grid search da salvare in file binario.

Funzioni:
    train_random_forest(X_train, y_train): Addestra un modello Random Forest Classifier con grid search.
    main(): Carica il dataset, esegue l'oversampling utilizzando SMOTENC, addestra un modello Random Forest Classifier utilizzando la funzione train_random_forest, e restituisce il miglior modello addestrato.

Moduli esterni richiesti:
    - sys
    - functions: Modulo contenente funzioni di supporto per la pulizia dei dati e l'oversampling.
    - sklearn.ensemble.RandomForestClassifier
    - sklearn.model_selection.GridSearchCV
"""
import sys
sys.path.append('../base_lib')
import functions as func

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
    """
    Addestra un modello Random Forest Classifier con grid search.

    Args:
        X_train (DataFrame): Features di addestramento.
        y_train (Series): Target di addestramento.

    Returns:
        RandomForestClassifier: Miglior modello della grid search.
    """
    model = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [3, 4, 5],
        'max_depth': [8],
        'criterion': ["gini", "entropy"],
        'min_samples_split': [2, 4, 6],
        'min_impurity_decrease': [0.0, 0.01, 0.02]
    }

    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=5,
                               scoring='f1_macro')
    
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

def main():
    """
    Carica il dataset, esegue l'oversampling utilizzando SMOTENC, addestra un modello Random Forest Classifier utilizzando la funzione train_random_forest, e restituisce il miglior modello addestrato.

    Returns:
        model: Modello addestrato per essere salvato in un file binario..
    """
    dataset = func.load_csv()

    X = dataset.drop(['LUX_01'], axis=1)
    y = dataset['LUX_01']

    df = func.oversampling(dataset, X, y)

    dataset = func.drop_cols(dataset)
    df = func.drop_cols(df)

    training_set, testing_set = func.train_test(dataset, df, False)

    feature_cols = ['BREED', 'GENDER_01', 'AGEATSURGERYmo', 'BODYWEIGHTKG', 'Taglia', 'BCS', 
                    'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE', 'ALO', 'CUPRETROVERSION', 'STEMANTEVERSIONREAL', 
                    'RECTUSFEMORISM.RELEASE', 'LUX_CR']

    X_train = training_set[feature_cols]
    y_train = training_set['LUX_01']

    model = train_random_forest(X_train, y_train)
    return model
