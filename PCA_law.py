import torch
import torch.nn as nn

def _require_same(x: torch.Tensor, ref: torch.Tensor, name="X"):
    if x.device != ref.device or x.dtype != ref.dtype:
        raise TypeError(
            f"{name} must be on device={ref.device}, dtype={ref.dtype} "
            f"(got device={x.device}, dtype={x.dtype}). Move all data once before calling."
        )

class PCA_law(nn.Module):
    def __init__(self, n_components, center=False, dtype=torch.float64):
        super().__init__()
        self.n_components = int(n_components)
        self.center = center
        # Buffers are created after fit; init placeholders for .to() compatibility
        self.register_buffer("mu", None, persistent=True)
        self.register_buffer("eigenvalues", None, persistent=True)
        self.register_buffer("eigenvectors", None, persistent=True)
        self._dtype = dtype

    @property
    def dtype(self):
        # dtype of eigenvectors if set, else desired default
        return self.eigenvectors.dtype if self.eigenvectors is not None else self._dtype

    @property
    def device(self):
        # device of eigenvectors if set, else CPU (changes after first .to())
        return self.eigenvectors.device if self.eigenvectors is not None else torch.device("cpu")

    def fit(self, X: torch.Tensor):
        # Expect X already on the desired device/dtype
        if X.dtype != self._dtype:
            raise TypeError(f"X dtype must be {self._dtype}, got {X.dtype}. Cast once before fit.")
        # center?
        mu = X.mean(dim=0) if self.center else torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
        Xred = X - mu
        C = Xred.T @ Xred
        evals, evecs = torch.linalg.eigh(C)  # ascending
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx]
        evecs = evecs[:, idx]

        self.register_buffer("mu", mu, persistent=True)
        self.register_buffer("eigenvalues", evals, persistent=True)
        self.register_buffer("eigenvectors", evecs, persistent=True)
        return self

    def compress(self, X: torch.Tensor):
        _require_same(X, self.eigenvectors, "X")
        return (X - self.mu) @ self.eigenvectors[:, :self.n_components]

    def reconstruct(self, ycomp: torch.Tensor):
        _require_same(ycomp, self.eigenvectors, "ycomp")
        return ycomp @ self.eigenvectors[:, :self.n_components].T + self.mu

    def classify(self, X: torch.Tensor, ord=2):
        _require_same(X, self.eigenvectors, "X")
        resid = (X - self.mu) @ self.eigenvectors[:, self.n_components:]
        return torch.linalg.vector_norm(resid, ord=ord, dim=1)

    def chi2(self, X: torch.Tensor, ep=1e-10):
        _require_same(X, self.eigenvectors, "X")
        Z = (X - self.mu) @ self.eigenvectors     # (n, d)
        inv_l = 1.0 / (self.eigenvalues + Z.new_tensor(ep))
        return (Z.pow(2) * inv_l).sum(dim=1)


class PCA_classifier(nn.Module):
    def __init__(self, n_components, center=False, dtype=torch.float64):
        super().__init__()
        self.n_components = int(n_components)
        self.center = center
        self.dtype_default = dtype
        self.yclasses = None
        self.laws = nn.ModuleList()

    @property
    def device(self):
        # device of the module parameters/buffers (after .to())
        return next(self.parameters(), torch.tensor([], device="cpu")).device

    @property
    def dtype(self):
        return next(self.parameters(), torch.tensor([], dtype=self.dtype_default)).dtype

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        # Expect both already on same device/dtype
        if X.device != y.device:
            raise TypeError("X and y must be on the same device.")
        if X.dtype != self.dtype_default:
            raise TypeError(f"X dtype must be {self.dtype_default}, got {X.dtype}. Cast once before fit.")
        if y.dtype not in (torch.int64, torch.int32):
            raise TypeError("y must be integer (torch.long/int64 recommended).")

        yclasses = torch.unique(y).sort().values
        self.yclasses = yclasses  # kept as plain tensor; moves with .to() if registered as buffer
        # Register as buffer so it follows .to()
        self.register_buffer("yclasses_buf", yclasses.to(device=X.device), persistent=True)

        self.laws = nn.ModuleList()
        for yv in yclasses:
            XX = X[y == yv]
            law = PCA_law(self.n_components, center=self.center, dtype=X.dtype).to(X.device, X.dtype)
            law.fit(XX)
            self.laws.append(law)
        return self

    def predict(self, X: torch.Tensor, ord=2, return_numpy=True):
        _require_same(X, self.laws[0].eigenvectors, "X")
        dists = torch.stack([law.classify(X, ord=ord) for law in self.laws], dim=0)  # (n_class, n)
        idx = torch.argmin(dists, dim=0)
        preds = self.yclasses_buf[idx]
        return preds.detach().cpu().numpy() if return_numpy else preds

    def predict_chi2(self, X: torch.Tensor, ep=1e-10, return_numpy=True):
        _require_same(X, self.laws[0].eigenvectors, "X")
        chi2s = torch.stack([law.chi2(X, ep=ep) for law in self.laws], dim=0)
        idx = torch.argmin(chi2s, dim=0)
        preds = self.yclasses_buf[idx]
        return preds.detach().cpu().numpy() if return_numpy else preds
