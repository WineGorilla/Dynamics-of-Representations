import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import os
from glob import glob
from tqdm import tqdm

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

    b = np.linalg.pinv(Phi_s) @ X.mean(axis=0) #求出均值在稳定空间中的稳定residual

    x = (Phi_s @ b)

    if center:
        x = x + mu[:, 0]  # 把均值加回去得到多个层所表示的稳定embedding

    return x.real.astype(np.float32)


def fuse_layers_dmd(X, r=1, k=1, eps=1e-8, center=True):
    X = np.asarray(X, dtype=np.float32)
    L, D = X.shape
    if L < 2:
        return X[0].copy()

    X1 = X[:-1].T
    X2 = X[1:].T

    if center:
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

    idx = np.argsort(np.abs(np.abs(eigvals) - 1.0))[:max(1, int(k))]
    Phi_s = Phi[:, idx]

    b = np.linalg.pinv(Phi_s) @ (X - X.mean(axis=0)).mean(axis=0) 

    x = (Phi_s @ b)

    if center:
        x = x + X.T[:, 0] 
    
    return x.real.astype(np.float32)





def fuse_all_audio_models(
    root="data/audio",
    k=3
):
    design_root = os.path.join(root, "design_matrix")
    fused_root  = os.path.join(root, "design_matrix_dmd_mean")

    os.makedirs(fused_root, exist_ok=True)

    # === 遍历每一个模型文件夹 ===
    model_names = sorted(
        d for d in os.listdir(design_root)
        if os.path.isdir(os.path.join(design_root, d))
    )

    print(f"Found {len(model_names)} audio models")

    for model in model_names:
        print(f"\n[Model] {model}")

        in_dir  = os.path.join(design_root, model)
        out_dir = os.path.join(fused_root, model)
        os.makedirs(out_dir, exist_ok=True)

        npy_files = sorted(glob(os.path.join(in_dir, "*.npy")))
        print(f"  Files: {len(npy_files)}")

        for in_path in tqdm(npy_files):
            fname = os.path.basename(in_path)
            out_path = os.path.join(out_dir, fname)

            X = np.load(in_path)      # (L, T, d)
            if X.ndim != 3:
                print(f"Skip {fname}, shape={X.shape}")
                continue

            L, T, d = X.shape
            fused = np.zeros((T, d), dtype=np.float32)

            for t in range(T):
                fused[t] = fuse_layers_single_time_dmd(X[:, t, :],r=k)

            np.save(out_path, fused)

fuse_all_audio_models(
    root="data/audio",
    k=1
)




