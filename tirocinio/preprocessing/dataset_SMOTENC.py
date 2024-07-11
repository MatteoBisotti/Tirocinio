import pandas as pd

import sys
sys.path.append("../base_lib")
import functions as func
sys.path.append("../Oversampling")
import smotenc as smo

dataset = func.load_csv()

X = dataset.drop(['LUX_01'], axis=1)
y = dataset['LUX_01']

dataset_oversampling = smo.oversampling(dataset, X, y)
dataset_oversampling.to_csv("../csv/dataset_SMOTENC.csv", index=False)