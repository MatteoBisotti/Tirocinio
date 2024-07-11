import pandas as pd

import sys
sys.path.append("../base_lib")
import functions as func
sys.path.append("../Oversampling")
import encoder as enc

dataset = func.load_csv()

categorical_features = [dtype.name == 'int64' for dtype in dataset.dtypes]

augmented_df = enc.encoder(dataset, categorical_features)
augmented_df.to_csv("../csv/dataset_encoder.csv", index=False)