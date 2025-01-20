"""
Modulo contenente la classe RandomForestModel per la gestione di modelli Random Forest.

Classi:
    RandomForestModel: Classe per la gestione di modelli Random Forest.
"""

from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd

class RandomForestModel(BaseModel):
    """
    Classe per la gestione di modelli RandomForest.
    """

    def __init__(self, n_estimators, max_depth):
        """
        Inizializza il modello RandomForestClassifier.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            random_state=42) 
        
    def get_estimator(self):
        """
        Restituisce il primo stimatore della random forest.
        """
        return self.model.estimators_[0]
        
    def print_tree(self, feature_cols):
        """
        Stampa il primo albero di decisione della random forest.
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
        Restituisce la feature importance del primo stimatore della random forest.
        """
        return self.get_estimator().feature_importances_
    
    def graph_feature_importance(self, feature_name):
        """
        Traccia il grafico della feature importance.
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
