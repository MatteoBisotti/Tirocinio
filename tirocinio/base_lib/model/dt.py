"""
Modulo contenente la classe DecisionTreeModel per la gestione di modelli di alberi decisionali.

Classi:
    DecisionTreeModel: Classe per la gestione di modelli di alberi decisionali, includendo funzioni
                       per addestrare, prevedere, e calcolare metriche di valutazione del modello.

Funzioni:
    __init__(self, max_depth, min_sample_split, min_impurity_decrease, criterion): Inizializza il modello DecisionTreeClassifier.
    print_tree(self, feature_cols): Traccia l'albero di decisione.
    feature_importance(self): Ritorna l'importanza delle feature del modello.
    graph_feature_importance(self, feature_name): Traccia un grafico dell'importanza delle feature.

Moduli esterni richiesti:
    sklearn.tree: Fornisce la classe DecisionTreeClassifier e funzioni per tracciare alberi decisionali.
    base_model: Modulo contenente la classe base BaseModel da cui ereditare.
    matplotlib: Fornisce un'API per tracciare grafici in Python.
    pandas: Fornisce strutture dati e strumenti di analisi per il linguaggio di programmazione Python.
"""

from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd

class DecisionTreeModel(BaseModel):
    """
    Classe per la gestione di modelli di alberi decisionali.

    Metodi:
        __init__(self, max_depth, min_sample_split, min_impurity_decrease, criterion): Inizializza il modello DecisionTreeClassifier.
        print_tree(self, feature_cols): Traccia l'albero di decisione.
        feature_importance(self): Ritorna l'importanza delle feature del modello.
        graph_feature_importance(self, feature_name): Traccia un grafico dell'importanza delle feature.
    """

    def __init__(self, max_depth, min_sample_split, min_impurity_decrease, criterion):
        """
        Inizializza il modello DecisionTreeClassifier.

        Args:
            max_depth (int): La profondità massima dell'albero.
            min_sample_split (int): Il numero minimo di campioni richiesti per suddividere un nodo interno.
            min_impurity_decrease (float): La quantità minima di riduzione dell'impurità richiesta per effettuare una suddivisione.
            criterion (str): La funzione di misurazione della qualità di una suddivisione (ad es. 'gini' o 'entropy').
        """
        self.model = DecisionTreeClassifier(random_state=42, 
                                            max_depth=max_depth, 
                                            min_samples_split=min_sample_split, 
                                            min_impurity_decrease=min_impurity_decrease,
                                            criterion=criterion)

    def print_tree(self, feature_cols):
        """
        Traccia l'albero di decisione.

        Args:
            feature_cols (list): Lista dei nomi delle feature.
        """
        plt.figure(figsize=(16, 12))
        plot_tree(decision_tree=self.model, 
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
        return self.model.feature_importances_

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
