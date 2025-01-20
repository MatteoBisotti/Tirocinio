"""
Modulo contenente la classe astratta BaseModel per la gestione di modelli di machine learning.

Classi:
    BaseModel: Classe astratta per la gestione di modelli di machine learning, includendo funzioni
               per addestrare, prevedere, e calcolare metriche di valutazione del modello.
"""

from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display


class BaseModel(ABC):
    """
    Classe astratta per la gestione di modelli di machine learning.
    """

    def train(self, X_train, y_train):
        """
        Addestra il modello con i dati di addestramento.

        Args:
            X_train (DataFrame): Dati x di addestramento.
            y_train (Series) ati y di addestramento.
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

    def statistics(self, X_test, y_test):
        """
        Calcola e ritorna le metriche di valutazione del modello in un DataFrame.
        Calcola la matrice di confusione e la stampa.
        Stampa il grafico a barre delle metriche con il metodo 'plot_matrics'

        Args:
            X_test (DataFrame): Dati di test.
            y_test (Series): Label dati di test.

        Returns:
            DataFrame: DataFrame contenente le metriche di valutazione del modello.
        """
        predictions = self.predict(X_test)

        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm.ravel()  

        print(cm)

        print("tn = ", tn)
        print("fp = ", fp)
        print("fn = ", fn)
        print("tp = ", tp)

        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions, average='macro', zero_division=1)
        precision = precision_score(y_test, predictions, average='macro', zero_division=1)
        f1 = f1_score(y_test, predictions, average='macro', zero_division=1)
        roc_auc = roc_auc_score(y_test, predictions)
        specificity = tn / (tn + fp)

        data = {
            'Metrica': ['Accuracy', 'Recall', 'Precision', 'F1-score', 'ROC AUC', 'Specificity'],
            'Valore': [accuracy, recall, precision, f1, roc_auc, specificity]
        }
        metrics_df = pd.DataFrame(data)
        display(metrics_df)
        self.plot_metrics(metrics_df)
        
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
            
        #barplot.grid(True, axis='y')
        #barplot.grid(False, axis='x')

        plt.xlabel('Metriche')
        plt.ylabel('Valori')
        plt.ylim(0, 1.05)
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