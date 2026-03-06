import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import re
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
# =========================
# 工具函数
# =========================

def natural_sort_key(s):
    match = re.search(r"run-(\d+)", s)
    return int(match.group(1)) if match else 9999


def align_fused(emb, fmri):
    T = min(emb.shape[0], fmri.shape[0])
    return emb[:T], fmri[:T]


def zscore(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


# =========================
# 稳定 Ridge
# =========================

def ridge_fit_stable(X, Y, alpha=50.0, jitter=1e-8):
    alpha = float(max(alpha, 1e-6))
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
# Pearson
# =========================

def fused_roi_pearson(X, Y, alpha=50.0):

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

def run_one_lang_model_fused(model_name, alpha=50.0):

    FMRI_ROOT = "data/lang/fmri"
    EMB_DIR   = f"data/lang/design_matrix_dmd_mean/{model_name}"
    SAVE_ROOT = f"results/dmd_mean/hrf/lang_fused/{model_name}"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    subs = sorted(s for s in os.listdir(FMRI_ROOT) if s.startswith("sub-"))
    group_results = []

    for subj in subs:
        print(f"\n===== Subject {subj} | Model: {model_name} =====")
        subj_dir = os.path.join(FMRI_ROOT, subj)

        run_files = sorted(
            [f for f in os.listdir(subj_dir) if f.endswith("_bold_shared.npy")],
            key=natural_sort_key
        )

        pearsons_subj = []

        for idx, fmri_fname in enumerate(run_files[:9], start=1):

            fmri_path = os.path.join(subj_dir, fmri_fname)
            emb_path = os.path.join(
                EMB_DIR,
                f"lppEN_section{idx}_bold_embedding.npy"
            )

            if not os.path.exists(emb_path):
                continue

            emb  = np.load(emb_path).astype(np.float32)
            fmri = np.load(fmri_path).astype(np.float32)
            #fmri = fmri[2:, :]

            if fmri.ndim != 2 or fmri.shape[1] != STANDARD_N_ROI:
                continue

            if emb.ndim != 2:
                raise ValueError(f"{model_name} wrong emb shape {emb.shape}")
            emb = apply_hrf_to_embedding(emb, tr=2.0)
            emb, fmri = align_fused(emb, fmri)

            if emb.shape[0] < 10:
                continue

            pearson = fused_roi_pearson(emb, fmri, alpha=alpha)
            pearsons_subj.append(pearson)

        if pearsons_subj:
            subj_mean = np.mean(np.stack(pearsons_subj), axis=0)
            group_results.append(subj_mean)

    if group_results:
        group_mean = np.mean(np.stack(group_results), axis=0)
        np.save(os.path.join(SAVE_ROOT, "group_mean_roi.npy"), group_mean)
        print(f"[DONE] {model_name}, alpha={alpha}, shape={group_mean.shape}")
    else:
        print(f"[SKIP] {model_name}")

    return model_name


# =========================
# 并行执行
# =========================

lang_models = [
    "albert-base-v2",
    "albert-large-v2",
    "bert-base-cased",
    "bert-base-multilingual-cased",
    "bert-base-uncased",
    "bert-large-cased",
    "bert-large-uncased",
    "deberta-base",
    "deberta-large",
    "distilbert-base-uncased",
    "electra-base-discriminator",
    "electra-large-discriminator",
    "roberta-base",
    "roberta-large",
    "xlm-roberta-base",
    "xlm-roberta-large",
]


if __name__ == "__main__":

    MAX_WORKERS = 6
    ALPHA = 10.0   # 别再用0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_one_lang_model_fused, m, ALPHA): m
            for m in lang_models
        }

        for f in as_completed(futures):
            model = futures[f]
            try:
                f.result()
            except Exception as e:
                print(f"[ERROR] {model}: {e}")