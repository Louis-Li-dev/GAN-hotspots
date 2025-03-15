from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

class RandomForestManager:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            oob_score=True,
            n_jobs=-1
        )

    def train(self, features, targets, verbose=1):
 
        
        self.model.fit(features, targets)
        train_mse = self.model.score(features, targets)
        self.model.oob_score_
        if verbose:
            print(f"Train MSE: {train_mse:.4f}")

    def predict(self, X, inverse_transformer=None):
        pred = self.model.predict(X)
        
        if inverse_transformer is not None:
            pred = inverse_transformer.predict(pred, inverse=True)
        return pred
