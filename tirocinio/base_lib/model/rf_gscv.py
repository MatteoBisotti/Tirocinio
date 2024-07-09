"""
Modulo contenente la classe RandomForestGscvModel per la gestione di modelli RandomForest con ricerca di griglia (GridSearchCV).

Classi:
    RandomForestGscvModel: Classe per la gestione di modelli RandomForest con ricerca di griglia,
                           includendo funzioni per addestrare, prevedere, e calcolare metriche di valutazione del modello.

Funzioni:
    __init__(self, param_grid, cv, scoring): Inizializza il modello GridSearchCV con RandomForestClassifier.
    best_estimator(self): Restituisce il miglior stimatore trovato dalla ricerca di griglia.
    get_best_params(self): Restituisce i migliori parametri trovati dalla ricerca di griglia.
    predict(self, X_test): Prevede i valori di X_test utilizzando il miglior stimatore.
    print_tree(self, feature_cols): Stampa il primo albero di decisione della foresta.
    feature_importance(self): Restituisce l'importanza delle caratteristiche.
    graph_feature_importance(self, feature_name): Traccia un grafico delle importanze delle caratteristiche.
    get_result(self): Restituisce i risultati della ricerca di griglia.

Moduli esterni richiesti:
    sklearn.model_selection: Fornisce la classe GridSearchCV per la ricerca di iperparametri.
    sklearn.ensemble: Fornisce la classe RandomForestClassifier per la classificazione RandomForest.
    base_model: Modulo contenente la classe base BaseModel da cui ereditare.
    matplotlib.pyplot: Per la creazione di grafici.
    sklearn.tree: Fornisce la funzione plot_tree per la visualizzazione degli alberi di decisione.
    pandas: Fornisce la struttura dati DataFrame per la gestione dei dati tabulari.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd

class RandomForestGscvModel(BaseModel):
    """
    Classe per la gestione di modelli RandomForest con ricerca di griglia.

    Metodi:
        __init__(self, param_grid, cv, scoring): Inizializza il modello GridSearchCV con RandomForestClassifier.
        best_estimator(self): Restituisce il miglior stimatore trovato dalla ricerca di griglia.
        get_best_params(self): Restituisce i migliori parametri trovati dalla ricerca di griglia.
        predict(self, X_test): Prevede i valori di X_test utilizzando il miglior stimatore.
        print_tree(self, feature_cols): Stampa il primo albero di decisione della foresta.
        feature_importance(self): Restituisce l'importanza delle caratteristiche.
        graph_feature_importance(self, feature_name): Traccia un grafico delle importanze delle caratteristiche.
    """

    def __init__(self, param_grid, cv, scoring):
        """
        Inizializza il modello GridSearchCV con RandomForestClassifier.

        Args:
            param_grid (dict): Dizionario contenente i parametri da cercare in GridSearchCV.
            cv (int): Numero di fold per la validazione incrociata.
            scoring (str): Funzione di valutazione da utilizzare nella ricerca di griglia.
        """
        self.model = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                                  param_grid=param_grid,
                                  cv=cv,
                                  scoring=scoring)

    def best_estimator(self):
        """
        Restituisce il miglior stimatore trovato dalla ricerca di griglia.

        Returns:
            RandomForestClassifier: Il miglior stimatore RandomForestClassifier.
        """
        return self.model.best_estimator_
    
    def get_best_params(self):
        """
        Restituisce i migliori parametri trovati dalla ricerca di griglia.

        Returns:
            pd.DataFrame: I migliori parametri trovati dalla ricerca di griglia.
        """
        best_params = self.model.best_params_
        results = {**best_params}
        results_df = pd.DataFrame([results])

        return results_df
    
    def predict(self, X_test):
        """
        Prevede i valori di X_test utilizzando il miglior stimatore.

        Args:
            X_test (array-like): I dati di test da prevedere.

        Returns:
            array: Le previsioni per X_test.
        """
        return self.best_estimator().predict(X_test)
    
    def print_tree(self, feature_cols):
        """
        Stampa il primo albero di decisione della foresta.

        Args:
            feature_cols (list): Lista dei nomi delle caratteristiche.
        """
        plt.figure(figsize=(16, 12))
        plot_tree(decision_tree=self.model.best_estimator_[0], 
                  feature_names=feature_cols, 
                  filled=True, 
                  rounded=True, 
                  class_names=True, max_depth=2)
        plt.title("Albero di decisione")
        plt.show()

    def feature_importance(self):
        """
        Restituisce l'importanza delle caratteristiche.

        Returns:
            array: Importanza delle caratteristiche.
        """
        return self.best_estimator().feature_importances_

    def graph_feature_importance(self, feature_name):
        """
        Traccia un grafico delle importanze delle caratteristiche.

        Args:
            feature_name (list): Lista dei nomi delle caratteristiche.
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
