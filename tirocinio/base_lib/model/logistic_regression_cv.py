"""
Modulo contenente la classe LogisticRegressionCvModel per la gestione di modelli di regressione logistica con cross validation.

Classi:
    LogisticRegressionCvModel: Classe per la definizione di modelli di regressione logistica con cross validation.
"""

from sklearn.linear_model import LogisticRegressionCV
from .base_model import BaseModel

class LogisticRegressionCvModel(BaseModel):
    """
    Classe per la definzione di modelli di regressione logistica con cross validation.
    """

    def __init__(self, cv):
        """
        Inizializza il modello LogisticRegressionCV.
        """
        self.model = LogisticRegressionCV(cv=cv, 
                                          random_state=42,
                                          max_iter=5000)
