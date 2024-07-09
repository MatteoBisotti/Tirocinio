"""
Modulo contenente la classe LogisticRegressionCvModel per la gestione di modelli di regressione logistica con validazione incrociata.

Classi:
    LogisticRegressionCvModel: Classe per la gestione di modelli di regressione logistica, includendo funzioni
                               per addestrare, prevedere, e calcolare metriche di valutazione del modello.

Funzioni:
    __init__(self, cv): Inizializza il modello LogisticRegressionCV.

Moduli esterni richiesti:
    sklearn.linear_model: Fornisce la classe LogisticRegressionCV per la regressione logistica con validazione incrociata.
    base_model: Modulo contenente la classe base BaseModel da cui ereditare.
"""

from sklearn.linear_model import LogisticRegressionCV
from .base_model import BaseModel

class LogisticRegressionCvModel(BaseModel):
    """
    Classe per la gestione di modelli di regressione logistica con validazione incrociata.

    Metodi:
        __init__(self, cv): Inizializza il modello LogisticRegressionCV.
    """

    def __init__(self, cv):
        """
        Inizializza il modello LogisticRegressionCV.

        Args:
            cv (int): Numero di fold per la validazione incrociata.
        """
        self.model = LogisticRegressionCV(cv=cv, 
                                          random_state=42,
                                          max_iter=5000)
