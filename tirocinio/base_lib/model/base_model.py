"""
Modulo contenente la classe astratta BaseModel per la gestione di modelli di machine learning.

Classi:
    BaseModel: Classe astratta per la gestione di modelli di machine learning, includendo funzioni
               per addestrare, prevedere, e calcolare metriche di valutazione del modello.

Funzioni:
    train(X_train, y_train): Addestra il modello con i dati di addestramento.
    predict(X_test): Prevede i risultati usando i dati di test.
    get_report(X_test, y_test): Genera un report di classificazione per i dati di test.
    print_report(X_test, y_test): Stampa il report di classificazione per i dati di test.
    get_stats(X_test, y_test): Calcola e ritorna metriche di valutazione del modello.
    statistics(X_test, y_test): Calcola e ritorna le metriche di valutazione del modello in un DataFrame.
    plot_metrics(metrics_df): Traccia un grafico a barre delle metriche di valutazione del modello.

Moduli esterni richiesti:
    abc: Fornisce strumenti per definire classi astratte.
    sklearn.metrics: Fornisce funzioni per calcolare metriche di valutazione per modelli di machine learning.
    pandas: Fornisce strutture dati e strumenti di analisi per il linguaggio di programmazione Python.
    seaborn: Fornisce un'interfaccia ad alto livello per disegnare grafici statistici.
    matplotlib: Fornisce un'API per tracciare grafici in Python.
"""

from abc import ABC, abstractmethod
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class BaseModel(ABC):
    """
    Classe astratta per la gestione di modelli di machine learning.

    Metodi:
        train(X_train, y_train): Addestra il modello con i dati di addestramento.
        predict(X_test): Prevede i risultati usando i dati di test.
        get_report(X_test, y_test): Genera un report di classificazione per i dati di test.
        print_report(X_test, y_test): Stampa il report di classificazione per i dati di test.
        get_stats(X_test, y_test): Calcola e ritorna metriche di valutazione del modello.
        statistics(X_test, y_test): Calcola e ritorna le metriche di valutazione del modello in un DataFrame.
        plot_metrics(metrics_df): Traccia un grafico a barre delle metriche di valutazione del modello.
    """

    def train(self, X_train, y_train):
        """
        Addestra il modello con i dati di addestramento.

        Args:
            X_train (DataFrame): Dati x di addestramento.
            y_train (Series): Dati y di addestramento.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Prevede i risultati usando i dati di test.

        Args:
            X_test (DataFrame): Dati di test.

        Returns:
            array: Predizioni del modello.
        """
        return self.model.predict(X_test)
    
    def get_report(self, X_test, y_test):
        """
        Genera un report di classificazione per i dati di test.

        Args:
            X_test (DataFrame): Dati di test.
            y_test (Series): Etichette di test.

        Returns:
            dict: Report di classificazione.
        """
        predictions = self.predict(X_test=X_test)
        return classification_report(y_test, predictions, output_dict=True)

    def print_report(self, X_test, y_test):
        """
        Stampa il report di classificazione per i dati di test.

        Args:
            X_test (DataFrame): Dati di test.
            y_test (Series): Etichette di test.
        """
        print("Report di classificazione:")
        print(self.get_report(X_test, y_test))

    def get_stats(self, X_test, y_test):
        """
        Calcola e ritorna metriche di valutazione del modello.

        Args:
            X_test (DataFrame): Dati di test.
            y_test (Series): Etichette di test.

        Returns:
            tuple: Tuple contenente accuratezza, precisione, richiamo, F1 score, e ROC AUC.
        """
        predictions = self.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)

        return accuracy, precision, recall, f1, roc_auc
    

    def statistics(self, X_test, y_test):
        """
        Calcola e ritorna le metriche di valutazione del modello in un DataFrame.

        Args:
            X_test (DataFrame): Dati di test.
            y_test (Series): Etichette di test.

        Returns:
            DataFrame: DataFrame contenente le metriche di valutazione del modello.
        """
        predictions = self.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions, average='macro', zero_division=1)
        precision = precision_score(y_test, predictions, average='macro', zero_division=1)
        f1 = f1_score(y_test, predictions, average='macro', zero_division=1)
        roc_auc = roc_auc_score(y_test, predictions)

        data = {
            'Metrica': ['Accuratezza', 'Sensibilità', 'Specificità', 'F1-score', 'ROC AUC'],
            'Valore': [accuracy, recall, precision, f1, roc_auc]
        }
        metrics_df = pd.DataFrame(data)
        return metrics_df

    def plot_metrics(self, metrics_df):
        """
        Traccia un grafico a barre delle metriche di valutazione del modello.

        Args:
            metrics_df (DataFrame): DataFrame contenente le metriche di valutazione del modello.
        """
        plt.figure(figsize=(12, 8))
        barplot = sns.barplot(x='Metrica', y='Valore', data=metrics_df, capsize=0.2, palette='deep', errorbar=None)

        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.2f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center',
                            xytext = (0, 9),
                            textcoords = 'offset points')
            
        barplot.grid(True, axis='y')
        barplot.grid(False, axis='x')

        plt.xlabel('Metriche')
        plt.ylabel('Valori')
        plt.title('Metriche del modello sui dati originali')
        plt.ylim(0, 1)
        plt.show()

    def graph_feature_importance(self, feature_name):
        """
        Traccia un grafico dell'importanza delle feature.

        Args:
            feature_name (list): Lista dei nomi delle feature.
        """
        importance = self.feature_importance()
        
        feature_importance = pd.DataFrame({'Feature': feature_name, 'Importance': importance})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])

        plt.title("Importanza delle feature")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()