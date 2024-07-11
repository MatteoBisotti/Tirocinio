import pandas as pd
import logging

import sys
sys.path.append("../Imputation")
import imputation as imp 

sys.path.append("../base_lib")
import models
import functions as func

from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(message)s')

file_handler = logging.FileHandler('../logs/dt_model_encoder.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger()

logger.handlers = []
logger.addHandler(file_handler)

def metrics_boxplot(metrics_total_df):
    """
    Generates and displays box plots for various metrics.

    Args:
        metrics_total_df (DataFrame): DataFrame containing metrics data.

    """
    # Melt the DataFrame for easier plotting with seaborn
    metrics_boxplot_melted = metrics_total_df.melt(var_name='Metric', value_name='Value')

    # Plot the box plots for each metric
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Metric', y='Value', data=metrics_boxplot_melted)
    plt.title('Box Plots delle Metriche')
    plt.ylabel('Value')
    plt.show()

# Creazione del grafico
def plot_metrics_mean_dv(summary_df):
    """
    Generates and displays a bar plot showing mean and standard deviation for each metric.

    Args:
        summary_df (DataFrame): DataFrame containing the summary statistics (mean and standard deviation) for each metric.

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
    Main function to execute the data processing, model training, and evaluation workflow.

    Workflow:
        - Load the dataset.
        - Perform oversampling and drop unnecessary columns.
        - Display the first few rows of the processed DataFrame.
        - Plot the outcome feature distribution before and after oversampling.
        - Train a decision tree model multiple times and collect metrics.
        - Display and plot the collected metrics.

    """

    dataset = pd.read_csv("../csv/dataset_original.csv")
    df = pd.read_csv("../csv/dataset_encoder.csv")

    dataset = func.drop_cols(dataset)
    df = func.drop_cols(df)

    display(df.head(5))

    func.plot_outcome_feature(dataset, 'LUX_01')
    func.plot_outcome_feature(df, 'LUX_01')

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []

    feature_cols = ['BREED', 'GENDER_01', 'AGEATSURGERYmo', 'BODYWEIGHTKG', 'Taglia', 'BCS', 
                    'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE', 'ALO', 'CUPRETROVERSION', 'STEMANTEVERSIONREAL', 
                    'RECTUSFEMORISM.RELEASE', 'LUX_CR']
    
    last_model = None


    for i in range(10):
        training_set, testing_set = func.train_test(dataset, df, True)

        X_train = training_set[feature_cols]
        y_train = training_set['LUX_01']

        X_test = testing_set[feature_cols]
        y_test = testing_set['LUX_01']

        model = models.decision_tree_model(X_train, X_test, 
                                        y_train, y_test,
                                        8,
                                        4,
                                        0.0,
                                        "gini")
        
        metrics_df = model.statistics(X_test, y_test)
        accuracies.append(metrics_df['Valore'][0])
        precisions.append(metrics_df['Valore'][1])
        recalls.append(metrics_df['Valore'][2])
        f1_scores.append(metrics_df['Valore'][3])
        roc_aucs.append(metrics_df['Valore'][4])

        logging.info(f"iterazione{i+1}:accuracy:{metrics_df['Valore'][0]}:precision:{metrics_df['Valore'][1]}:recall:{metrics_df['Valore'][2]}:f1_score:{metrics_df['Valore'][3]}:roc_auc:{metrics_df['Valore'][4]}")

        last_model = model

    
    metrics_total_df = pd.DataFrame({
        'Accuratezza': accuracies,
        'Specificità': precisions,
        'Sensibilità': recalls,
        'F1 Score': f1_scores,
        'ROC AUC': roc_aucs
    })
    display(metrics_total_df)
    metrics_boxplot(metrics_total_df)

    means = metrics_total_df[1:].mean()
    std_devs = metrics_total_df[1:].std()
    summary_df = pd.DataFrame({
        'Metrica': means.index,
        'Media': means.values,
        'Deviazione Standard': std_devs.values
    })
    display(summary_df)
    plot_metrics_mean_dv(summary_df)

    for index, row in summary_df.iterrows():
        logging.info(f"{row['Metrica']}:{row['Media']}:{row['Deviazione Standard']}")

    last_model.print_tree(feature_cols)
    last_model.graph_feature_importance(feature_cols)