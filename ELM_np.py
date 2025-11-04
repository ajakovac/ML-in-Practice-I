import numpy as np

# --- 🌙 ELM class ---
class ELM:
    def __init__(self, n_hidden=50, activation=np.tanh):
        self.n_hidden = n_hidden
        self.activation = activation

    def fit(self, X, y):
        self.W = np.random.randn(X.shape[1], self.n_hidden)
        self.b = np.random.randn(self.n_hidden)
        H = self.activation(X @ self.W + self.b)
        self.beta = np.linalg.pinv(H) @ y

    def predict(self, X):
        H = self.activation(X @ self.W + self.b)
        return H @ self.beta