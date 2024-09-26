import pandas as pd
import sys

sys.path.append("../base_lib")
import functions as func

sys.path.append("../Oversampling")
import encoder_dummy as enc_dummy

dataset = pd.read_csv("../csv/dataset_original.csv")

final_dataset = enc_dummy.encoder(dataset)

final_dataset.to_csv("../csv/dataset_encoder.csv", index=False)