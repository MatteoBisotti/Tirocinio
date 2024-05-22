from abc import ABC, abstractmethod
import numpy as np

class ImputerNr(ABC):

    def __init__(self, dataset):
        self.dataset = dataset

    @abstractmethod
    def build_model(self, input_dim):
        pass

    def impute_missing_value(self, dataset, feature):
        # Crea una copia del DataFrame
        dataset_copy = dataset.copy()

        # Divide i dati in feature e target
        X = dataset_copy.drop([feature], axis=1) # contiene tutte le colonne del DataFrame tranne la feature da imputare.
        y = dataset_copy[feature]                # contiene solo la colonna della feature da imputare

        # Identifica i valori mancanti nella feature da imputare
        missing_bool = y.isna()             # contiene valori booleani, true se il dato è mancante e false altrimenti

        # Separa i dati con e senza valori mancanti
        X_train = X[~missing_bool]                  # contiene le righe di X dove i valori non sono mancanti
        y_train = y[~missing_bool]                  # contiene le righe di y dove i valori non sono mancanti
        X_missing_values = X[missing_bool]          # contiene le righe di X dove i valori in y sono mancanti 
        
        # Costruzione e addestramento del modello
        model = self.build_model(X_train.shape[1])       # gli passo in input il numero di feature di X_train, in questo caso 23 -> 24 - feature da imputare
        model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)    # si allena il modello su X_train e y_train per 100 epoche
        
        # Predizione dei valori mancanti
        y_pred = model.predict(X_missing_values)    # il modello addestrato viene utilizzato per predire i valori mancanti di y usando X_missing
        y_pred = np.argmax(y_pred, axis=1)

        # Imputazione dei valori mancanti    
        y.loc[missing_bool] = y_pred.flatten()

        return y

