import pandas as pd
import sys

sys.path.append("../base_lib")
import functions as func

sys.path.append("../Oversampling")
import smotenc_dummy as smo_dummy

dataset = pd.read_csv("../csv/dataset_dummy_feature.csv")

X = dataset.drop(['LUX_01'], axis=1)
y = dataset['LUX_01']

dataset_oversampling = smo_dummy.oversampling(dataset, X, y)

dataset_oversampling.to_csv("../csv/dataset_SMOTENC_dummy.csv", index=False)