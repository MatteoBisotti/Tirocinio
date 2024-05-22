from abc import ABC, abstractmethod
import pandas as pd

class ImputerKNN(ABC):

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    @abstractmethod
    def get_knn_imputer(self):
        pass

    def impute_value(self, dataset):
        imputer = self.get_knn_imputer()
        dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
        return dataset