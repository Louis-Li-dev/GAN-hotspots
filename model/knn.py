from sklearn.neighbors import NearestNeighbors
import numpy as np


from sklearn.neighbors import NearestNeighbors
import numpy as np

class KNNManager:
    def __init__(self, n_neighbors=5, algorithm='auto'):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm)

    def train(self, features, targets):
        self.model.fit(features)
        self.targets = targets

    def predict(self, X, inverse_transformer=None):
        indices = self.model.kneighbors(X, return_distance=False)
        pred = self.targets[indices]
        if inverse_transformer is not None:
            pred = inverse_transformer.predict(pred, inverse=True)
        return pred

class KNNFeaturesExtractor:
    def __init__(
            self,
            n_neighbors=3,
            leaf_size=100
        ):
        
        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(
            leaf_size=leaf_size,
            n_jobs=-1
        )
    def fit(self, x):
        self.knn.fit(x)
        self.x = x
    def predict(self, x):
        return_array = []
        for ele in x:
            indices = self.knn.kneighbors(
                ele.reshape(1, -1), 
                n_neighbors=self.n_neighbors,
                return_distance=False
            )[0]
            return_array.append(self.x[indices])
        return np.array(return_array)
