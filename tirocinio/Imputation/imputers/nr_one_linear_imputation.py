import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from .imputer_nr import ImputerNr

class LinearNrImputer(ImputerNr):

    def build_model(self, input_dim):
        model = Sequential([
            # primo strato che prende in input il numero di feature del dataset, il numero di neuroni, la funzione di attivazione ReLU
            Dense(10, activation='relu', input_dim=input_dim),      

            # strato di dropout, che disattiva casualmente un numero di neuroni durante ogni passaggio di addestramento per prevenire overfitting
            Dropout(0.2),                                           

            # strato che prende in input il numero di neuroni e la funzione di attivazione ReLU
            Dense(5, activation='relu'),

            # strato di dropout
            Dropout(0.2),

            # strato finale con un singolo neurone e una funzione di attivazione lineare
            Dense(1, activation='linear')  # Predice una singola feature
        ])

        # compilazione del modello in cui si specifica la funzione di perdita (MSE) e si utilizza come ottimizzatore Adam
        model.compile(loss='mse', optimizer='adam')
        return model