from sklearn.linear_model import LogisticRegressionCV
from .base_model import BaseModel

class LogisticRegressionCvModel(BaseModel):

    def __init__(self, cv):
        self.model = LogisticRegressionCV(cv=cv, 
                                          random_state=42,
                                          max_iter=5000)