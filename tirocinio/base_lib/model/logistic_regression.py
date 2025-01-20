"""
Modulo contenente la classe LogisticRegressionModel per la gestione di modelli di regressione logistica.

Classi:
    LogisticRegressionModel: Classe per la definizione di modelli di regressione logistica.
"""

from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """
    Classe per la gestione di modelli di regressione logistica.
    """

    def __init__(self):
        """
        Inizializza il modello LogisticRegression.
        """
        super().__init__()  
        self.model = LogisticRegression(max_iter=5000, random_state=42)
