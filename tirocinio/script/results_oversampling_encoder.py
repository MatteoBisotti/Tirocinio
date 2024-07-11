import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("../base_lib")
import functions as func

from IPython.display import display

def main():
    dataset = pd.read_csv("../csv/dataset_encoder.csv")
    display(dataset.head(5))

    dataset = func.drop_cols(dataset)

    func.display_corr_matrix(dataset)

    func.plot_outcome_feature(dataset, 'LUX_01')
