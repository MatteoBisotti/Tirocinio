"""
Modulo contenente la classe DecisionTreeGscvModel per la gestione di modelli di alberi decisionali
con ottimizzazione dei parametri tramite GridSearchCV.

Classi:
    DecisionTreeGscvModel: Classe per la gestione di modelli di alberi decisionali, includendo funzioni
                           per addestrare, prevedere, e calcolare metriche di valutazione del modello,
                           con ottimizzazione dei parametri tramite GridSearchCV.

Funzioni:
    __init__(self, param_grid, cv, scoring): Inizializza il modello con GridSearchCV.
    best_estimator(self): Ritorna il miglior stimatore dopo l'ottimizzazione.
    get_best_params(self): Ritorna i migliori parametri trovati tramite GridSearchCV.
    predict(self, X_test): Prevede i risultati usando il miglior stimatore.
    print_tree(self, feature_cols): Traccia l'albero di decisione.
    feature_importance(self): Ritorna l'importanza delle feature del modello.
    graph_feature_importance(self, feature_name): Traccia un grafico dell'importanza delle feature.

Moduli esterni richiesti:
    sklearn.model_selection: Fornisce strumenti per la suddivisione dei dati e la ricerca di parametri ottimali.
    sklearn.tree: Fornisce la classe DecisionTreeClassifier e funzioni per tracciare alberi decisionali.
    base_model: Modulo contenente la classe base BaseModel da cui ereditare.
    matplotlib: Fornisce un'API per tracciare grafici in Python.
    pandas: Fornisce strutture dati e strumenti di analisi per il linguaggio di programmazione Python.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd

class DecisionTreeGscvModel(BaseModel):
    """
    Classe per la gestione di modelli di alberi decisionali con ottimizzazione dei parametri tramite GridSearchCV.

    Metodi:
        __init__(self, param_grid, cv, scoring): Inizializza il modello con GridSearchCV.
        best_estimator(self): Ritorna il miglior stimatore dopo l'ottimizzazione.
        get_best_params(self): Ritorna i migliori parametri trovati tramite GridSearchCV.
        predict(self, X_test): Prevede i risultati usando il miglior stimatore.
        print_tree(self, feature_cols): Traccia l'albero di decisione.
        feature_importance(self): Ritorna l'importanza delle feature del modello.
        graph_feature_importance(self, feature_name): Traccia un grafico dell'importanza delle feature.
    """

    def __init__(self, param_grid, cv, scoring):
        """
        Inizializza il modello con GridSearchCV.

        Args:
            param_grid (dict): Dizionario contenente i parametri da ottimizzare.
            cv (int): Numero di fold per la validazione incrociata.
            scoring (str): Metodologia di scoring per valutare le performance del modello.
        """
        self.model = GridSearchCV(DecisionTreeClassifier(random_state=42),
                                  param_grid=param_grid,
                                  cv=cv,
                                  scoring=scoring)

    def best_estimator(self):
        """
        Ritorna il miglior stimatore dopo l'ottimizzazione.

        Returns:
            DecisionTreeClassifier: Miglior modello dopo l'ottimizzazione.
        """
        return self.model.best_estimator_
    
    def get_best_params(self):
        """
        Ritorna i migliori parametri trovati tramite GridSearchCV.

        Returns:
            DataFrame: DataFrame contenente i migliori parametri.
        """
        best_params = self.model.best_params_
        results = {**best_params}
        results_df = pd.DataFrame([results])

        return results_df

    def predict(self, X_test):
        """
        Prevede i risultati usando il miglior stimatore.

        Args:
            X_test (DataFrame): Dati di test.

        Returns:
            array: Predizioni del modello.
        """
        return self.best_estimator().predict(X_test)
    
    def print_tree(self, feature_cols):
        """
        Traccia l'albero di decisione.

        Args:
            feature_cols (list): Lista dei nomi delle feature.
        """
        plt.figure(figsize=(16, 12))
        plot_tree(decision_tree=self.best_estimator(), 
                  feature_names=feature_cols, 
                  filled=True, 
                  rounded=True, 
                  class_names=True, max_depth=2)
        plt.title("Albero di decisione")
        plt.show()

    def feature_importance(self):
        """
        Ritorna l'importanza delle feature del modello.

        Returns:
            array: Importanza delle feature.
        """
        return self.best_estimator().feature_importances_

    def graph_feature_importance(self, feature_name):
        """
        Traccia un grafico dell'importanza delle feature.

        Args:
            feature_name (list): Lista dei nomi delle feature.
        """
        importance = self.feature_importance()

        feature_importance = pd.DataFrame({'Feature': feature_name, 'Importance': importance})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])

        plt.title("Importanza delle feature")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()
