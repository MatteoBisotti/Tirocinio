from sklearn.impute import KNNImputer
from .imputer_knn import ImputerKNN

class KnnDistanceImputer(ImputerKNN):

    def get_knn_imputer(self):
        return KNNImputer(n_neighbors=self.n_neighbors, weights='distance')