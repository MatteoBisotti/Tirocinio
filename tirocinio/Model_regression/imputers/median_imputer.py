import numpy as np
from .imputer import Imputer

class MedianImputer(Imputer):

    def impute_value(self):
        return self.dataset[self.name_feature].median()