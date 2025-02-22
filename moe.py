import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

class MOE(BaseEstimator):
    def __init__(self, n_experts=3):
        self.n_experts = n_experts
        self.gating = KMeans(n_clusters=n_experts)
        self.experts = [LinearRegression() for _ in range(n_experts)]
        
    def fit(self, X, y):
        # Train gating network (clustering)
        self.gating.fit(X)
        clusters = self.gating.predict(X)
        
        # Train expert networks
        for i in range(self.n_experts):
            mask = (clusters == i)
            if np.sum(mask) > 0:
                self.experts[i].fit(X[mask], y[mask])
        
        return self
    
    def predict(self, X):
        # Get cluster assignments
        clusters = self.gating.predict(X)
        
        # Initialize predictions
        predictions = np.zeros(X.shape[0])
        
        # Get predictions from appropriate experts
        for i in range(self.n_experts):
            mask = (clusters == i)
            if np.sum(mask) > 0:
                predictions[mask] = self.experts[i].predict(X[mask])
        
        return predictions

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    
    # Create and train MOE model
    model = MOE(n_experts=3)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    print("Predictions shape:", y_pred.shape)