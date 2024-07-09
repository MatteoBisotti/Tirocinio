import pandas as pd

import sys
sys.path.append("../base_lib")
import functions as func

dataset = func.load_csv()

dataset.to_csv("../csv/dataset_original.csv", index=False)