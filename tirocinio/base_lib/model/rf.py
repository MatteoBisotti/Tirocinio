"""
Modulo contenente la classe RandomForestModel per la gestione di modelli RandomForest.

Classi:
    RandomForestModel: Classe per la gestione di modelli RandomForest,
                       includendo funzioni per addestrare, prevedere, e calcolare metriche di valutazione del modello.

Funzioni:
    __init__(self, n_estimators, max_depth): Inizializza il modello RandomForestClassifier.
    get_estimator(self): Restituisce il primo stimatore della foresta.
    print_tree(self, feature_cols): Stampa il primo albero di decisione della foresta.
    feature_importance(self): Restituisce l'importanza delle caratteristiche.
    graph_feature_importance(self, feature_name): Traccia un grafico delle importanze delle caratteristiche.

Moduli esterni richiesti:
    sklearn.ensemble: Fornisce la classe RandomForestClassifier per la classificazione RandomForest.
    base_model: Modulo contenente la classe base BaseModel da cui ereditare.
    matplotlib.pyplot: Per la creazione di grafici.
    sklearn.tree: Fornisce la funzione plot_tree per la visualizzazione degli alberi di decisione.
    pandas: Fornisce la struttura dati DataFrame per la gestione dei dati tabulari.
"""

from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd

class RandomForestModel(BaseModel):
    """
    Classe per la gestione di modelli RandomForest.

    Metodi:
        __init__(self, n_estimators, max_depth): Inizializza il modello RandomForestClassifier.
        get_estimator(self): Restituisce il primo stimatore della foresta.
        print_tree(self, feature_cols): Stampa il primo albero di decisione della foresta.
        feature_importance(self): Restituisce l'importanza delle caratteristiche.
        graph_feature_importance(self, feature_name): Traccia un grafico delle importanze delle caratteristiche.
    """

    def __init__(self, n_estimators, max_depth):
        """
        Inizializza il modello RandomForestClassifier.

        Args:
            n_estimators (int): Numero di alberi nella foresta.
            max_depth (int): Profondità massima degli alberi.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            random_state=42) 
        
    def get_estimator(self):
        """
        Restituisce il primo stimatore della foresta.

        Returns:
            DecisionTreeClassifier: Il primo albero di decisione della foresta.
        """
        return self.model.estimators_[0]
        
    def print_tree(self, feature_cols):
        """
        Stampa il primo albero di decisione della foresta.

        Args:
            feature_cols (list): Lista dei nomi delle caratteristiche.
        """
        plt.figure(figsize=(16, 12))
        plot_tree(decision_tree=self.get_estimator(), 
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
        return self.get_estimator().feature_importances_
    
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
