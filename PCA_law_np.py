import numpy as np

class PCA_law:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mu = 0 #np.mean(X, axis=0)
        Xred = X-self.mu
        self.C = Xred.T@Xred
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:,idx]
        return self
    
    def compress(self, X):
        return (X-self.mu)@self.eigenvectors[:, :self.n_components]
    
    def reconstruct(self, ycomp):
        return ycomp@self.eigenvectors[:, :self.n_components].T + self.mu
    
    def classify(self, X, ord=2):
        return np.linalg.norm((X-self.mu)@self.eigenvectors[:, self.n_components:], axis=1, ord=ord)
    
    def chi2(self, X, ep=1e-10):
        return np.diag((X-self.mu)@self.eigenvectors@np.diag(1/(self.eigenvalues + ep))@self.eigenvectors.T@(X-self.mu).T)

class PCA_classifier:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        self.yclasses = sorted(set(y))
        self.laws = [ PCA_law(self.n_components).fit(XX) for XX in [ X[y==yval] for yval in self.yclasses]]
    
    def predict(self, X, ord=2):
        return np.array([self.yclasses[k] for k in np.argmin(np.array([self.laws[i].classify(X, ord) for i in range(len(self.yclasses))]), axis=0)])

    def predict_chi2(self, X, ep=1e-10):
        return [self.yclasses[k] for k in np.argmin(np.array([self.laws[i].chi2(X,ep) for i in range(len(self.yclasses))]), axis=0)]