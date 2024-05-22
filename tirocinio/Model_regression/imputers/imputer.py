from abc import ABC, abstractmethod

class Imputer(ABC):

    def __init__(self, dataset, name_feature):
        self.dataset = dataset
        self.name_feature = name_feature

    @abstractmethod
    def impute_value(self):
        pass

    def impute(self):
        metric = self.impute_value()
        self.dataset[self.name_feature] = self.dataset[self.name_feature].fillna(metric)
        return self.dataset
