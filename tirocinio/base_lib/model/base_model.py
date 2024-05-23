from abc import ABC, abstractmethod
from sklearn.metrics import classification_report

class BaseModel(ABC):

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def print_report(self, X_test, y_test):
        predictions = self.predict(X_test=X_test)
        print("Report di classificazione:")
        print(classification_report(y_test, predictions))