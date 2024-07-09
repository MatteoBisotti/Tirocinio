import pandas as pd

import sys
sys.path.append("../base_lib")
import functions as func

dataset = func.load_csv()

binary_features = ['GENDER_01', 'LUX_CR', 'Taglia']

augmented_df = func.encoder(dataset, binary_features)
augmented_df.to_csv("../csv/dataset_encoder.csv", index=False)