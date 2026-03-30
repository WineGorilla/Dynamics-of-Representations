import numpy as np

def fuse_layers_single_time_dmd(X, r=1, k=1, eps=1e-8, center=True):
    X = np.asarray(X, dtype=np.float32)
    L, D = X.shape
    if L < 2:
        return X[0].copy()

    X1 = X[:-1].T
    X2 = X[1:].T

    if center:
        mu = X1.mean(axis=1, keepdims=True)
        X1 = X1 - mu
        X2 = X2 - mu  # 用同一个均值做对齐

    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    r = int(min(max(1, r), S.size))
    U = U[:, :r]
    S = S[:r]
    V = Vt[:r, :].T

    invS = 1.0 / (S + eps)
    A = U.T @ ((X2 @ V) * invS)      # 等价写法
    eigvals, W = np.linalg.eig(A)

    Phi = ((X2 @ V) * invS) @ W

    idx = np.argsort(np.abs(np.abs(eigvals) - 1.0))[:max(1, int(k))]
    Phi_s = Phi[:, idx]

    b = np.linalg.pinv(Phi_s) @ X.mean(axis=0)

    x = (Phi_s @ b)

    # if center:
    #     x = x + mu[:, 0]  # 把均值加回去

    return x.real.astype(np.float32)



def fuse_layers_single_dmd(X, r=1, k=1, eps=1e-8, center=True):
    X = np.asarray(X, dtype=np.float32)
    L, D = X.shape
    if L < 2:
        return X[0].copy()

    X1 = X[:-1].T
    X2 = X[1:].T

    if center:
        mu = X1.mean(axis=1, keepdims=True)
        X1 = X1 - mu
        X2 = X2 - mu  # 用同一个均值做对齐

    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    r = int(min(max(1, r), S.size))
    U = U[:, :r]
    S = S[:r]
    V = Vt[:r, :].T

    invS = 1.0 / (S + eps)
    A = U.T @ ((X2 @ V) * invS)      # 等价写法
    eigvals, W = np.linalg.eig(A)

    Phi = ((X2 @ V) * invS) @ W

    idx = np.argsort(np.abs(np.abs(eigvals) - 1.0))[:max(1, int(k))]
    Phi_s = Phi[:, idx]

    b = np.linalg.pinv(Phi_s) @ X.mean(axis=0)

    x = (Phi_s @ b)

    return x.real.astype(np.float32)



import numpy as np


def fuse_layers_single_soft_dmd(
    X,
    r=3,
    center=1.0,
    sigma=0.1,
    eps=1e-8,
    center_data=True
):
    X = np.asarray(X, dtype=np.float32)
    L, D = X.shape

    if L < 2:
        return X[0].copy()
    X1 = X[:-1].T
    X2 = X[1:].T

    if center_data:
        mu = X1.mean(axis=1, keepdims=True)
        X1 = X1 - mu
        X2 = X2 - mu

    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    r = int(min(max(1, r), S.size))
    U = U[:, :r]
    S = S[:r]
    V = Vt[:r, :].T

    invS = 1.0 / (S + eps)

    A = U.T @ ((X2 @ V) * invS)

    eigvals, W = np.linalg.eig(A)

    Phi = ((X2 @ V) * invS) @ W

    b = np.linalg.pinv(Phi) @ X.mean(axis=0)

    # spectral radius
    rho = np.abs(eigvals)
    weights = np.exp(-((rho - center) ** 2) / (2 * sigma ** 2))

    weights = weights / (weights.sum() + eps)

    x = Phi @ (weights * b)

    return x.real.astype(np.float32)








def choose_rank(S, threshold=0.95):
    energy = np.cumsum(S**2) / np.sum(S**2)
    r = np.searchsorted(energy, threshold) + 1
    return r


def compute_dmd_eigenvalues(X, r=None, energy_threshold=0.9, eps=1e-8, center=True):

    X = np.asarray(X, dtype=np.float32)
    L, D = X.shape

    if L < 2:
        return None

    X1 = X[:-1].T
    X2 = X[1:].T

    if center:
        mu = X1.mean(axis=1, keepdims=True)
        X1 = X1 - mu
        X2 = X2 - mu

    # SVD
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)

    # 自动选 rank
    if r is None:
        r = choose_rank(S, energy_threshold)

    # 限制 r 不超过理论最大
    r = min(r, S.size)

    U = U[:, :r]
    S = S[:r]
    V = Vt[:r, :].T

    invS = 1.0 / (S + eps)

    # Reduced DMD operator
    A = U.T @ ((X2 @ V) * invS)

    eigvals, _ = np.linalg.eig(A)

    return eigvals





# 重建法
def fuse_layers_single_reconstruct_dmd(
    X,
    r=12,
    center=1.0,
    sigma=0.3,
    eps=1e-8,
    center_data=True
):
    X = np.asarray(X, dtype=np.float32)
    L, D = X.shape

    if L < 2:
        return X[0].copy()

    # build snapshot matrices
    X1 = X[:-1].T
    X2 = X[1:].T

    if center_data:
        mu = X1.mean(axis=1, keepdims=True)
        X1 = X1 - mu
        X2 = X2 - mu

    # SVD
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    r = int(min(max(1, r), S.size))
    U = U[:, :r]
    S = S[:r]
    V = Vt[:r, :].T

    invS = 1.0 / (S + eps)

    # low-rank A
    A = U.T @ ((X2 @ V) * invS)

    # eigendecomposition
    eigvals, W = np.linalg.eig(A)
    eigvals = eigvals / np.maximum(1.0, np.abs(eigvals))
    # DMD modes
    Phi = ((X2 @ V) * invS) @ W

    # initial amplitudes (use first layer)
    b = np.linalg.pinv(Phi) @ X[0]

    # spectral radius weighting
    rho = np.abs(eigvals)
    weights = np.exp(-((rho - center) ** 2) / (2 * sigma ** 2))
    weights = weights / (weights.sum() + eps)

    # mode amplitudes with weighting
    b_weighted = b * weights

    # reconstruct all layers
    X_rec = []
    for l in range(L):
        lambda_power = eigvals ** l
        x_l = Phi @ (b_weighted * lambda_power)
        X_rec.append(x_l.real)

    X_rec = np.stack(X_rec, axis=0)

    # layer fusion (mean pooling)
    x_model = X_rec.mean(axis=0)

    return x_model.astype(np.float32)