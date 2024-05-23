from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel

class DecisionTreeModel(BaseModel):

    def __init__(self, max_depth):

        self.model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)