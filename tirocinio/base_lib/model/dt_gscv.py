"""
Modulo contenente la classe DecisionTreeGscvModel per la gestione di modelli di alberi decisionali
con ottimizzazione dei parametri tramite GridSearchCV.

Classi:
    DecisionTreeGscvModel: Classe per la gestione di modelli di alberi decisionali, includendo funzioni
                           per addestrare, prevedere, e calcolare metriche di valutazione del modello,
                           con ottimizzazione dei parametri tramite GridSearchCV.
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
    """

    def __init__(self, param_grid, cv, scoring):
        """
        Inizializza il modello con decision tree con grid search.
        """
        self.model = GridSearchCV(DecisionTreeClassifier(random_state=42),
                                  param_grid=param_grid,
                                  cv=cv,
                                  scoring=scoring)

    def best_estimator(self):
        """
        Ritorna il miglior stimatore.
        """
        return self.model.best_estimator_
    
    def get_best_params(self):
        """
        Ritorna i migliori parametri trovati tramite GridSearchCV.
        """
        best_params = self.model.best_params_
        results = {**best_params}
        results_df = pd.DataFrame([results])

        return results_df

    def predict(self, X_test):
        """
        Prevede i risultati usando il miglior stimatore.
        """
        return self.best_estimator().predict(X_test)
    
    def print_tree(self, feature_cols):
        """
        Traccia l'albero di decisione del miglior stimatore.
        """
        plt.figure(figsize=(12, 8))
        plot_tree(decision_tree=self.best_estimator(), 
                  feature_names=feature_cols, 
                  filled=True, 
                  rounded=True, 
                  class_names=True, max_depth=2)
        plt.show()

    def feature_importance(self):
        """
        Ritorna la feature importance del miglior stimatore.
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
