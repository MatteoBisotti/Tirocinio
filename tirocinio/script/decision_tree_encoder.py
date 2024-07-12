"""
Questo script esegue il caricamento, la elaborazione e la valutazione di un dataset utilizzando un albero decisionale. 
Include il sovracampionamento dei dati, la visualizzazione delle metriche e il salvataggio dei risultati.

Funzioni:
    - metrics_boxplot: Genera e visualizza box plot per varie metriche.
    - plot_metrics_mean_dv: Genera e visualizza un grafico a barre mostrando la media e la deviazione standard per ciascuna metrica.
    - main: Funzione principale che esegue il workflow di elaborazione dei dati, addestramento del modello e valutazione.

Moduli esterni richiesti:
    - pandas
    - logging
    - sys
    - imputation (da Imputation): Modulo personalizzato per la imputazione dei dati mancanti.
    - models (da base_lib): Modulo personalizzato che contiene i modelli di machine learning.
    - functions (da base_lib): Funzioni di utilità per la gestione del dataset.
    - IPython.display
    - matplotlib.pyplot
    - seaborn
"""

import pandas as pd
import logging
import sys
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../Imputation")
import imputation as imp 

sys.path.append("../base_lib")
import models
import functions as func

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
file_handler = logging.FileHandler('../logs/dt_model_encoder.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(message)s'))
logger = logging.getLogger()
logger.handlers = []
logger.addHandler(file_handler)

def metrics_boxplot(metrics_total_df):
    """
    Genera e visualizza box plot per varie metriche.

    Args:
        metrics_total_df (DataFrame): DataFrame contenente i dati delle metriche.

    """
    # Trasforma il DataFrame per una più facile creazione del plot con seaborn
    metrics_boxplot_melted = metrics_total_df.melt(var_name='Metric', value_name='Value')

    # Crea i box plot per ciascuna metrica
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Metric', y='Value', data=metrics_boxplot_melted)
    plt.title('Box Plots delle Metriche')
    plt.ylabel('Value')
    plt.show()

def plot_metrics_mean_dv(summary_df):
    """
    Genera e visualizza un grafico a barre mostrando la media e la deviazione standard per ciascuna metrica.

    Args:
        summary_df (DataFrame): DataFrame contenente le statistiche (media e deviazione standard) per ciascuna metrica.

    """
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x='Metrica', y='Media', data=summary_df, capsize=0.2)
    barplot.errorbar(summary_df.index, summary_df['Media'], yerr=summary_df['Deviazione Standard'], fmt='none', c='black', capsize=5)

    # Personalizzazione della griglia per mostrare solo le linee orizzontali
    barplot.grid(True, axis='y')
    barplot.grid(False, axis='x')

    plt.title('Media e Deviazione Standard per Metrica')
    plt.xlabel('Metrica')
    plt.ylabel('Media')
    plt.ylim(0, 1)
    plt.show()

def main():
    """
    Funzione principale per eseguire il workflow di elaborazione dei dati, addestramento del modello e valutazione.

    Workflow:
        - Carica il dataset.
        - Esegue il sovracampionamento e elimina le colonne non necessarie.
        - Visualizza le prime righe del DataFrame elaborato.
        - Crea grafici della distribuzione delle etichette prima e dopo il sovracampionamento.
        - Addestra un modello ad albero decisionale più volte e raccoglie le metriche.
        - Visualizza e crea grafici delle metriche raccolte.

    """
    # Carica il dataset originale e sovracampionato
    dataset = pd.read_csv("../csv/dataset_original.csv")
    df = pd.read_csv("../csv/dataset_encoder.csv")

    # Elimina le colonne non necessarie
    dataset = func.drop_cols(dataset)
    df = func.drop_cols(df)

    # Visualizza le prime 5 righe del DataFrame elaborato
    display(df.head(5))

    # Crea grafici della distribuzione delle etichette prima e dopo il sovracampionamento
    func.plot_outcome_feature(dataset, 'LUX_01')
    func.plot_outcome_feature(df, 'LUX_01')

    # Liste per raccogliere le metriche
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []

    feature_cols = ['BREED', 'GENDER_01', 'AGEATSURGERYmo', 'BODYWEIGHTKG', 'Taglia', 'BCS', 
                    'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE', 'ALO', 'CUPRETROVERSION', 'STEMANTEVERSIONREAL', 
                    'RECTUSFEMORISM.RELEASE', 'LUX_CR']
    
    last_model = None

    # Addestramento e valutazione del modello per 10 iterazioni
    for i in range(10):
        training_set, testing_set = func.train_test(dataset, df, True)

        X_train = training_set[feature_cols]
        y_train = training_set['LUX_01']

        X_test = testing_set[feature_cols]
        y_test = testing_set['LUX_01']

        # Addestramento del modello ad albero decisionale
        model = models.decision_tree_model(X_train, X_test, 
                                           y_train, y_test,
                                           8,
                                           4,
                                           0.0,
                                           "gini")
        
        # Raccolta delle metriche
        metrics_df = model.statistics(X_test, y_test)
        accuracies.append(metrics_df['Valore'][0])
        precisions.append(metrics_df['Valore'][1])
        recalls.append(metrics_df['Valore'][2])
        f1_scores.append(metrics_df['Valore'][3])
        roc_aucs.append(metrics_df['Valore'][4])

        logging.info(f"iterazione{i+1}:accuracy:{metrics_df['Valore'][0]}:precision:{metrics_df['Valore'][1]}:recall:{metrics_df['Valore'][2]}:f1_score:{metrics_df['Valore'][3]}:roc_auc:{metrics_df['Valore'][4]}")

        last_model = model

    # Creazione di un DataFrame per le metriche raccolte
    metrics_total_df = pd.DataFrame({
        'Accuratezza': accuracies,
        'Specificità': precisions,
        'Sensibilità': recalls,
        'F1 Score': f1_scores,
        'ROC AUC': roc_aucs
    })
    display(metrics_total_df)
    metrics_boxplot(metrics_total_df)

    # Calcolo della media e della deviazione standard per ciascuna metrica
    means = metrics_total_df[1:].mean()
    std_devs = metrics_total_df[1:].std()
    summary_df = pd.DataFrame({
        'Metrica': means.index,
        'Media': means.values,
        'Deviazione Standard': std_devs.values
    })
    display(summary_df)
    plot_metrics_mean_dv(summary_df)

    # Log delle statistiche riassuntive
    for index, row in summary_df.iterrows():
        logging.info(f"{row['Metrica']}:{row['Media']}:{row['Deviazione Standard']}")

    # Stampa dell'albero decisionale e importanza delle feature
    last_model.print_tree(feature_cols)
    last_model.graph_feature_importance(feature_cols)

if __name__ == "__main__":
    main()