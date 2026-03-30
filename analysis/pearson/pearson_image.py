# final/vision.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

STANDARD_N_ROI = 200

from nilearn.glm.first_level import spm_hrf
from scipy.signal import fftconvolve
def apply_hrf_to_embedding(emb, tr=2.0):
    """
    emb: (T, D)
    return: (T, D) after HRF convolution
    """
    hrf = spm_hrf(tr)
    T, D = emb.shape
    emb_hrf = np.zeros_like(emb)

    for d in range(D):
        emb_hrf[:, d] = fftconvolve(
            emb[:, d],
            hrf,
            mode="full"
        )[:T]

    return emb_hrf

def align_fused(emb, fmri):
    """
    emb : (T, D)
    fmri: (T, V)
    """
    T = min(emb.shape[0], fmri.shape[0])
    return emb[:T, :], fmri[:T, :]


def zscore(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


def ridge_fit_stable(X, Y, alpha=50.0, jitter=1e-8):
    """
    Stable ridge for both D>T and D<=T.

    X: (T, D)
    Y: (T, V)
    returns W: (D, V)
    """
    # 关键：永远别让 alpha=0
    alpha = float(max(alpha, 1e-6))

    T, D = X.shape

    if T == 0:
        raise ValueError("T==0 after alignment; check your inputs.")

    if D > T:
        # dual form: W = X^T (XX^T + alpha I)^-1 Y
        K = X @ X.T  # (T, T)
        A = K + (alpha + jitter) * np.eye(T, dtype=X.dtype)
        # 用 solve，稳定；A 必须正定
        tmp = np.linalg.solve(A, Y)   # (T, V)
        W = X.T @ tmp                 # (D, V)
    else:
        # primal form: W = (X^T X + alpha I)^-1 X^T Y
        A = (X.T @ X) + (alpha + jitter) * np.eye(D, dtype=X.dtype)
        B = X.T @ Y
        W = np.linalg.solve(A, B)

    return W


def fused_roi_pearson(X, Y, alpha=50.0):
    """
    X: (T, D)
    Y: (T, V)
    return corr: (V,)
    """
    # 标准化（非常重要）
    Xs = zscore(X).astype(np.float64, copy=False)
    Ys = zscore(Y).astype(np.float64, copy=False)

    W = ridge_fit_stable(Xs, Ys, alpha=alpha)
    Yp = Xs @ W  # (T, V)

    # Pearson：因为已 zscore，corr = dot / (|| ||)
    num = np.sum(Ys * Yp, axis=0)
    den = np.sqrt(np.sum(Ys * Ys, axis=0)) * np.sqrt(np.sum(Yp * Yp, axis=0))
    corr = np.zeros(Ys.shape[1], dtype=np.float64)
    ok = den > 0
    corr[ok] = num[ok] / den[ok]
    corr[~np.isfinite(corr)] = 0.0
    return corr.astype(np.float32)


def run_one_model_fused(model_name, center, alpha=50.0):

    center_str = f"{int(center*10):02d}"

    FMRI_ROOT = "data/img/fmri"
    EMB_ROOT  = f"data/img/design_matrix_dmd_soft{center_str}/{model_name}"
    SAVE_ROOT = f"results/pearson/{center_str}/img_fused/{model_name}"

    os.makedirs(SAVE_ROOT, exist_ok=True)

    pearson_list = []

    subs = sorted([s for s in os.listdir(FMRI_ROOT) if s.startswith("sub-")])

    for subj in subs:

        subj_dir = os.path.join(FMRI_ROOT, subj)

        for ses in sorted([s for s in os.listdir(subj_dir) if s.startswith("ses-")]):

            ses_dir = os.path.join(subj_dir, ses)

            fmri_files = [
                f for f in os.listdir(ses_dir)
                if f.endswith("_bold_shared.npy")
            ]

            X_runs = []
            Y_runs = []

            for fname in fmri_files:

                fmri_path = os.path.join(ses_dir, fname)

                emb_path = os.path.join(
                    EMB_ROOT,
                    subj,
                    ses,
                    fname.replace("_bold_shared.npy", "_bold_embedding.npy")
                )

                if not os.path.exists(emb_path):
                    continue

                fmri = np.load(fmri_path).astype(np.float32)

                if fmri.ndim != 2 or fmri.shape[1] != STANDARD_N_ROI:
                    continue

                emb = np.load(emb_path).astype(np.float32)

                if emb.ndim != 2:
                    raise ValueError(
                        f"{model_name} got emb.ndim={emb.ndim}, expected 2. shape={emb.shape}"
                    )

                emb = apply_hrf_to_embedding(emb, tr=2.0)

                emb, fmri = align_fused(emb, fmri)

                if emb.shape[0] < 10:
                    continue

                X_runs.append(emb)
                Y_runs.append(fmri)

            # 如果这个 session 没数据就跳过
            if len(X_runs) == 0:
                continue

            # ===== concat runs =====
            X_session = np.concatenate(X_runs, axis=0)
            Y_session = np.concatenate(Y_runs, axis=0)

            pearson = fused_roi_pearson(
                X_session,
                Y_session,
                alpha=alpha
            )

            pearson_list.append(pearson)

    if pearson_list:

        group_mean = np.mean(
            np.stack(pearson_list, axis=0),
            axis=0
        )

        np.save(
            os.path.join(SAVE_ROOT, "group_mean_roi.npy"),
            group_mean
        )

        print(
            f"[DONE] {model_name}, alpha={alpha}, shape={group_mean.shape}"
        )

    else:

        print(f"[SKIP] {model_name}: no usable data.")

    return model_name


vision_models = [
    "beit-base-patch16-224-pt22k-ft22k",
    "beit-large-patch16-224-pt22k-ft22k",
    "data2vec-vision-base",
    "data2vec-vision-large",
    "deit-base-patch16-224",
    "deit-small-patch16-224",
    "dino-vitb16",
    "dino-vits16",
    "dinov2-base",
    "dinov2-large",
    "dinov2-small",
    "vit-base-patch16-224-in21k",
    "vit-large-patch16-224-in21k",
    "vit-mae-base",
    "vit-mae-large",
    "vit-msn-base",
    "vit-msn-large",
]


if __name__ == "__main__":
    MAX_WORKERS = 6
    ALPHA = 10.0  
    #centers = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4]
    centers = [1.0]
    for center in centers:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(run_one_model_fused, m,center, ALPHA): m for m in vision_models}

            for f in as_completed(futures):
                model = futures[f]
                try:
                    f.result()
                except Exception as e:
                    print(f"[ERROR] {model}: {e}")


