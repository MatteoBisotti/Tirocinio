from abc import ABC, abstractmethod

class ImputerRegression(ABC):

    def __init__(self, dataset, name_feature):
        self.dataset = dataset
        self.name_feature = name_feature

    @abstractmethod
    def get_model(self):
        pass

    def impute_value(self):
        dataset_notnull = self.dataset.dropna()
        dataset_null = self.dataset[self.dataset[self.name_feature].isnull()]

        X_train = dataset_notnull.drop([self.name_feature], axis=1)
        y_train = dataset_notnull[self.name_feature]

        model = self.get_model()
        model.fit(X_train, y_train)

        X_test = dataset_null.drop([self.name_feature], axis=1)
        predict_values = model.predict(X_test)

        self.dataset.loc[self.dataset[self.name_feature].isnull(), self.name_feature] = predict_values

        return self.dataset