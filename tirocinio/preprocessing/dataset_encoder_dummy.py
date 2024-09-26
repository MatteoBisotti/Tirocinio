import pandas as pd
import sys

sys.path.append("../base_lib")
import functions as func

sys.path.append("../Oversampling")
import encoder_dummy as enc_dummy

dataset = pd.read_csv("../csv/dataset_dummy_feature.csv")

final_dataset = enc_dummy.encoder(dataset)

final_dataset.to_csv("../csv/dataset_encoder_dummy.csv", index=False)