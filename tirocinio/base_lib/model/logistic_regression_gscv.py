"""
Modulo contenente la classe LogisticRegressionGscvModel per la gestione di modelli di regressione logistica con ricerca di griglia (GridSearchCV).

Classi:
    LogisticRegressionGscvModel: Classe per la gestione di modelli di regressione logistica con grid search.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionGscvModel(BaseModel):
    """
    Classe per la gestione di modelli di regressione logistica con grid search.
    """

    def __init__(self, param_grid, cv, scoring):
        """
        Inizializza il modello LogisticRegression con grid search.
        """
        self.model = GridSearchCV(estimator=LogisticRegression(random_state=42, max_iter=5000), 
                                  param_grid=param_grid, 
                                  cv=cv,
                                  scoring=scoring)
        
    def best_estimator(self):
        """
        Restituisce il miglior stimatore trovato dalla grid search.
        """
        return self.model.best_estimator_
    
    def print_best_params(self):
        """
        Restituisce i migliori parametri trovati dalla grid search.
        """
        return self.model.best_params_

    def predict(self, X_test):
        """
        Prevede i valori di X_test utilizzando il miglior stimatore.
        """
        return self.best_estimator().predict(X_test)