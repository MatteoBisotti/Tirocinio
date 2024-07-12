"""
Questo modulo contiene funzioni per l'addestramento di un modello Decision Tree Classifier da salvare in file binario.

Funzioni:
    train_decision_tree(X_train, y_train): Addestra un modello Decision Tree Classifier 
    main(): Carica il dataset, esegue l'oversampling utilizzando SMOTENC, prepara i dati, suddivide il dataset in set di addestramento e test, addestra un modello Decision Tree Classifier e restituisce il modello addestrato.

Moduli esterni richiesti:
    - sys
    - functions: Modulo contenente funzioni di supporto per la pulizia dei dati e l'oversampling.
    - sklearn.tree.DecisionTreeClassifier
"""

import sys
sys.path.append('../base_lib')
import functions as func

from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train):
    """
    Addestra un modello Decision Tree Classifier

    Args:
        X_train (DataFrame): Features di addestramento.
        y_train (Series): Target di addestramento.

    Returns:
        model: Modello addestrato.
    """
    model = DecisionTreeClassifier(max_depth=8, 
                                   min_samples_split=4, 
                                   min_impurity_decrease=0.0, 
                                   criterion='gini')
    
    model.fit(X_train, y_train)

    return model

def main():
    """
    Carica il dataset, esegue l'oversampling utilizzando SMOTENC, prepara i dati,
    suddivide il dataset in set di addestramento e test, addestra un modello
    Decision Tree Classifier e restituisce il modello addestrato.

    Returns:
        DecisionTreeClassifier: Modello addestrato per essere salvato in un file binario.
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

    model = train_decision_tree(X_train, y_train)
    return model
