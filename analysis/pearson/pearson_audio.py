import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from nilearn.glm.first_level import spm_hrf
from scipy.signal import fftconvolve

STANDARD_N_ROI = 200


# =========================
# 工具函数
# =========================

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

def extract_task_name(fname):
    m = re.search(r"task-([a-zA-Z0-9]+)", fname)
    return m.group(1) if m else None


def align_fused(emb, fmri):
    T = min(emb.shape[0], fmri.shape[0])
    return emb[:T], fmri[:T]


def zscore(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


# =========================
# 稳定 Ridge（关键）
# =========================

def ridge_fit_stable(X, Y, alpha=50.0, jitter=1e-8):
    """
    X: (T, D)
    Y: (T, V)
    """
    alpha = float(max(alpha, 1e-6))  # 永远别让 alpha=0

    T, D = X.shape

    if D > T:
        # dual form
        K = X @ X.T
        A = K + (alpha + jitter) * np.eye(T, dtype=X.dtype)
        tmp = np.linalg.solve(A, Y)
        W = X.T @ tmp
    else:
        # primal form
        A = X.T @ X + (alpha + jitter) * np.eye(D, dtype=X.dtype)
        B = X.T @ Y
        W = np.linalg.solve(A, B)

    return W


# =========================
# Pearson 计算
# =========================

def fused_roi_pearson(X, Y, alpha=10.0):

    Xs = zscore(X).astype(np.float64, copy=False)
    Ys = zscore(Y).astype(np.float64, copy=False)

    W = ridge_fit_stable(Xs, Ys, alpha=alpha)
    Yp = Xs @ W

    num = np.sum(Ys * Yp, axis=0)
    den = np.sqrt(np.sum(Ys**2, axis=0)) * np.sqrt(np.sum(Yp**2, axis=0))

    corr = np.zeros(Ys.shape[1], dtype=np.float64)
    ok = den > 0
    corr[ok] = num[ok] / den[ok]
    corr[~np.isfinite(corr)] = 0.0

    return corr.astype(np.float32)


# =========================
# 主函数
# =========================

def run_one_model_fused(model_name, center, alpha=50.0):
    center_str = f"{int(center*10):02d}"
    FMRI_ROOT = "data/audio/fmri"
    EMB_ROOT  = f"data/audio/design_matrix_dmd_soft{center_str}/{model_name}"
    SAVE_ROOT = f"results/pearson/{center_str}/audio_fused/{model_name}"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    pearson_list = []

    subs = sorted(s for s in os.listdir(FMRI_ROOT) if s.startswith("sub-"))

    for subj in subs:
        subj_dir = os.path.join(FMRI_ROOT, subj)
        sessions = sorted(s for s in os.listdir(subj_dir) if s.startswith("ses-"))

        for ses in sessions:
            ses_dir = os.path.join(subj_dir, ses)

            fmri_files = [
                f for f in os.listdir(ses_dir)
                if f.endswith("_bold_shared_roi.npy")
            ]

            for fname in fmri_files:

                task = extract_task_name(fname)
                if task is None:
                    continue

                fmri_path = os.path.join(ses_dir, fname)
                emb_path  = os.path.join(EMB_ROOT, f"{task}.npy")

                if not os.path.exists(emb_path):
                    continue

                fmri = np.load(fmri_path).astype(np.float32)
                #fmri = fmri[2:, :]  # 去前2 TR

                if fmri.ndim != 2 or fmri.shape[1] != STANDARD_N_ROI:
                    continue

                emb = np.load(emb_path).astype(np.float32)

                if emb.ndim != 2:
                    raise ValueError(
                        f"{model_name} embedding dim wrong: {emb.shape}"
                    )
                emb = apply_hrf_to_embedding(emb, tr=2.0)
                emb, fmri = align_fused(emb, fmri)

                if emb.shape[0] < 10:
                    continue

                pearson = fused_roi_pearson(emb, fmri, alpha=alpha)
                pearson_list.append(pearson)

    if pearson_list:
        group_mean = np.mean(np.stack(pearson_list), axis=0)
        np.save(os.path.join(SAVE_ROOT, "group_mean_roi.npy"), group_mean)
        print(f"[DONE] {model_name}, alpha={alpha}, shape={group_mean.shape}")
    else:
        print(f"[SKIP] {model_name}")

    return model_name


audio_models = [
    'data2vec-audio-base',
    'data2vec-audio-large',
    'hubert-base-ls960',
    'hubert-large-ls960-ft',
    'wav2vec2-base-960h',
    'wav2vec2-base-superb-ks',
    'wav2vec2-large-xlsr-53',
    'wav2vec2-xls-r-1b',
    'wav2vec2-xls-r-300m',
    'wavlm-base',
    'wavlm-base-plus',
    'wavlm-large',
]

if __name__ == "__main__":

    MAX_WORKERS = 6
    ALPHA = 10.0  
    #centers = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    centers = [1.0]
    for center in centers:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(run_one_model_fused, m, center, ALPHA): m
                for m in audio_models
            }

            for f in as_completed(futures):
                model = futures[f]
                try:
                    f.result()
                except Exception as e:
                    print(f"[ERROR] {model}: {e}")