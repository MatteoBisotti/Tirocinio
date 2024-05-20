from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):

    def __init__(self):
        super().__init__()  
        self.model = LogisticRegression(max_iter=5000, random_state=42)