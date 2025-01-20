"""
Modulo contenente la classe RandomForestGscvModel per la gestione di modelli RandomForest con GridSearchCV.

Classi:
    RandomForestGscvModel: Classe per la gestione di modelli RandomForest con grid search.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd

class RandomForestGscvModel(BaseModel):
    """
    Classe per la gestione di modelli RandomForest con grid search.
    """

    def __init__(self, param_grid, cv, scoring):
        """
        Inizializza il modello GridSearchCV con RandomForestClassifier.
        """
        self.model = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                                  param_grid=param_grid,
                                  cv=cv,
                                  scoring=scoring)

    def best_estimator(self):
        """
        Restituisce il miglior stimatore trovato dalla grid search.
        """
        return self.model.best_estimator_
    
    def get_best_params(self):
        """
        Restituisce i migliori parametri trovati dalla grid search.
        """
        best_params = self.model.best_params_
        results = {**best_params}
        results_df = pd.DataFrame([results])

        return results_df
    
    def predict(self, X_test):
        """
        Prevede i valori di X_test utilizzando il miglior stimatore.
        """
        return self.best_estimator().predict(X_test)
    
    def print_tree(self, feature_cols):
        """
        Stampa il primo albero di decisione del miglior stimatore.
        """
        plt.figure(figsize=(12, 8))
        plot_tree(decision_tree=self.model.best_estimator_[0], 
                  feature_names=feature_cols, 
                  filled=True, 
                  rounded=True, 
                  class_names=True, max_depth=2)
        plt.show()

    def feature_importance(self):
        """
        Restituisce la feature importance del miglior stimatore.
        """
        return self.best_estimator().feature_importances_

    def graph_feature_importance(self, feature_name):
        """
        Traccia il grafico della feature importance.
        """
        importance = self.feature_importance()

        feature_importance = pd.DataFrame({'Feature': feature_name, 'Importance': importance})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])

        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()
