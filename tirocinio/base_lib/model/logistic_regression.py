"""
Modulo contenente la classe LogisticRegressionModel per la gestione di modelli di regressione logistica.

Classi:
    LogisticRegressionModel: Classe per la gestione di modelli di regressione logistica, includendo funzioni
                             per addestrare, prevedere e calcolare metriche di valutazione del modello.

Funzioni:
    __init__(self): Inizializza il modello LogisticRegression.

Moduli esterni richiesti:
    sklearn.linear_model: Fornisce la classe LogisticRegression per la regressione logistica.
    base_model: Modulo contenente la classe base BaseModel da cui ereditare.
"""

from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """
    Classe per la gestione di modelli di regressione logistica.

    Metodi:
        __init__(self): Inizializza il modello LogisticRegression.
    """

    def __init__(self):
        """
        Inizializza il modello LogisticRegression.

        Args:
            max_iter (int): Numero massimo di iterazioni per l'ottimizzatore.
            random_state (int): Stato random per garantire la riproducibilità.
        """
        super().__init__()  
        self.model = LogisticRegression(max_iter=5000, random_state=42)
"""
Modulo contenente la classe LogisticRegressionModel per la gestione di modelli di regressione logistica.

Classi:
    LogisticRegressionModel: Classe per la gestione di modelli di regressione logistica, includendo funzioni
                             per addestrare, prevedere e calcolare metriche di valutazione del modello.

Funzioni:
    __init__(self): Inizializza il modello LogisticRegression.

Moduli esterni richiesti:
    sklearn.linear_model: Fornisce la classe LogisticRegression per la regressione logistica.
    base_model: Modulo contenente la classe base BaseModel da cui ereditare.
"""

from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """
    Classe per la gestione di modelli di regressione logistica.

    Metodi:
        __init__(self): Inizializza il modello LogisticRegression.
    """

    def __init__(self):
        """
        Inizializza il modello LogisticRegression.

        Args:
            max_iter (int): Numero massimo di iterazioni per l'ottimizzatore.
            random_state (int): Stato random per garantire la riproducibilità.
        """
        super().__init__()  
        self.model = LogisticRegression(max_iter=5000, random_state=42)
