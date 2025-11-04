# torch_elm.py
import torch
import torch.nn as nn

class ELM(nn.Module):
    """
    PyTorch version of your ELM (Extreme Learning Machine).
    - works fully on GPU if tensors are on CUDA
    - uses torch.linalg.pinv for Moore–Penrose pseudoinverse
    - accepts any activation: torch.tanh, torch.relu, etc.
    """
    def __init__(self, n_hidden=50, activation=torch.tanh, dtype=torch.float64):
        super().__init__()
        self.n_hidden = n_hidden
        self.activation = activation
        self.dtype = dtype
        self.W = None
        self.b = None
        self.beta = None

    def fit(self, X, y):
        """
        X: [N, d]  input tensor
        y: [N, c]  target tensor (can be 1D for regression)
        """
        device = X.device
        d = X.shape[1]
        N = X.shape[0]

        # initialize random hidden layer weights and biases
        self.W = torch.randn(d, self.n_hidden, dtype=self.dtype, device=device)
        self.b = torch.randn(self.n_hidden, dtype=self.dtype, device=device)

        # hidden activations
        H = self.activation(X @ self.W + self.b)  # [N, n_hidden]

        # solve for beta via pseudoinverse
        H_pinv = torch.linalg.pinv(H)
        self.beta = H_pinv @ y                    # [n_hidden, c]
        return self

    def predict(self, X):
        H = self.activation(X @ self.W + self.b)
        return H @ self.beta
