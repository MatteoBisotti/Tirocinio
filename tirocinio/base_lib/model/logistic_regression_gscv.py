"""
Modulo contenente la classe LogisticRegressionGscvModel per la gestione di modelli di regressione logistica con ricerca di griglia (GridSearchCV).

Classi:
    LogisticRegressionGscvModel: Classe per la gestione di modelli di regressione logistica con ricerca di griglia,
                                 includendo funzioni per addestrare, prevedere e calcolare metriche di valutazione del modello.

Funzioni:
    __init__(self, param_grid, cv, scoring): Inizializza il modello GridSearchCV con LogisticRegression.
    best_estimator(self): Restituisce il miglior stimatore trovato dalla ricerca di griglia.
    print_best_params(self): Restituisce i migliori parametri trovati dalla ricerca di griglia.
    predict(self, X_test): Prevede i valori di X_test utilizzando il miglior stimatore.

Moduli esterni richiesti:
    sklearn.model_selection: Fornisce la classe GridSearchCV per la ricerca di iperparametri.
    sklearn.linear_model: Fornisce la classe LogisticRegression per la regressione logistica.
    base_model: Modulo contenente la classe base BaseModel da cui ereditare.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionGscvModel(BaseModel):
    """
    Classe per la gestione di modelli di regressione logistica con ricerca di griglia.

    Metodi:
        __init__(self, param_grid, cv, scoring): Inizializza il modello GridSearchCV con LogisticRegression.
        best_estimator(self): Restituisce il miglior stimatore trovato dalla ricerca di griglia.
        print_best_params(self): Restituisce i migliori parametri trovati dalla ricerca di griglia.
        predict(self, X_test): Prevede i valori di X_test utilizzando il miglior stimatore.
    """

    def __init__(self, param_grid, cv, scoring):
        """
        Inizializza il modello GridSearchCV con LogisticRegression.

        Args:
            param_grid (dict): Dizionario contenente i parametri da cercare in GridSearchCV.
            cv (int): Numero di fold per la validazione incrociata.
            scoring (str): Funzione di valutazione da utilizzare nella ricerca di griglia.
        """
        self.model = GridSearchCV(estimator=LogisticRegression(random_state=42, max_iter=5000), 
                                  param_grid=param_grid, 
                                  cv=cv,
                                  scoring=scoring)
        
    def best_estimator(self):
        """
        Restituisce il miglior stimatore trovato dalla ricerca di griglia.

        Returns:
            LogisticRegression: Il miglior stimatore LogisticRegression.
        """
        return self.model.best_estimator_
    
    def print_best_params(self):
        """
        Restituisce i migliori parametri trovati dalla ricerca di griglia.

        Returns:
            dict: I migliori parametri trovati dalla ricerca di griglia.
        """
        return self.model.best_params_

    def predict(self, X_test):
        """
        Prevede i valori di X_test utilizzando il miglior stimatore.

        Args:
            X_test (array-like): I dati di test da prevedere.

        Returns:
            array: Le previsioni per X_test.
        """
        return self.best_estimator().predict(X_test)
