

from sklearn.decomposition import PCA

class PCATransformer:
    def __init__(
            self,
            n_components=.98
        ):
        self.pca = PCA(n_components=n_components)

    def fit(self, x):
        self.pca.fit(x)
    def fit_and_predict(self, x):
        self.pca.fit(x)
        return self.predict(x)
    def predict(self, x, inverse=False):
        return self.pca.transform(x) if inverse == False \
            else self.pca.inverse_transform(x)
