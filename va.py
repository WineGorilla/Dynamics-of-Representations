"""
Layer-wise Affine Fitting Validation
======================================
For each adjacent layer pair (k, k+1), fit:
  x_{k+1} = A_k @ x_k + c_k
using all stimuli via ridge regression.

Report per-layer R^2 and relative error, aggregate by modality.

Usage:
  CUDA_VISIBLE_DEVICES=2 python va.py
"""

import torch
import numpy as np
import os
from glob import glob
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def extract_layer_pairs_vision(data):
    """
    Vision format: (n_layers, n_tr, d)
    Extract non-zero TRs.
    """
    norms = np.linalg.norm(data, axis=2)
    active_mask = norms.sum(axis=0) > 0
    active_trs = np.where(active_mask)[0]

    valid = []
    for t in active_trs:
        traj = data[:, t, :]
        if np.any(np.linalg.norm(traj, axis=1) == 0):
            continue
        valid.append(t)

    if len(valid) == 0:
        return None
    return data[:, valid, :]  # (n_layers, N_valid, d)


def extract_layer_pairs_audio(data):
    """
    Audio format: (n_layers, n_chunks, d) — all chunks are valid samples.
    Filter out any chunk where any layer is exactly zero.
    """
    n_layers, n_chunks, d = data.shape
    layer_norms = np.linalg.norm(data, axis=2)  # (n_layers, n_chunks)
    valid_mask = np.all(layer_norms > 0, axis=0)  # (n_chunks,)

    if valid_mask.sum() == 0:
        return None
    return data[:, valid_mask, :]


def ridge_fit_gpu(X, Y, alpha=1.0):
    """
    Ridge regression: Y = X @ W + b
    X: (N, d_in), Y: (N, d_out)
    Returns: R^2, relative_error
    """
    N, d_in = X.shape
    d_out = Y.shape[1]

    # Center
    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # W = (X^T X + alpha I)^{-1} X^T Y
    XtX = Xc.T @ Xc + alpha * torch.eye(d_in, device=DEVICE, dtype=X.dtype)
    XtY = Xc.T @ Yc
    W = torch.linalg.solve(XtX, XtY)

    # Predict
    Y_hat = Xc @ W + Y_mean
    Y_pred_centered = Y_hat - Y_mean

    # R^2
    ss_res = torch.sum((Yc - Y_pred_centered) ** 2).item()
    ss_tot = torch.sum(Yc ** 2).item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    # Relative error: ||Y - Y_hat||_F / ||Y||_F
    residual = torch.norm(Y - Y_hat, 'fro').item()
    norm_y = torch.norm(Y, 'fro').item()
    rel_err = residual / (norm_y + 1e-12)

    return r2, rel_err


def validate_model_layerwise(model_dir, alpha=1.0, n_shuffle=10, fmt="vision"):
    """
    fmt: 'vision' for (n_layers, n_tr, d) with non-zero TR filtering
         'audio'  for (n_layers, n_chunks, d) where every chunk is a sample
    """
    if fmt == "vision":
        npy_files = sorted(glob(os.path.join(model_dir, "**",
                                              "*_bold_embedding.npy"),
                                 recursive=True))
        extractor = extract_layer_pairs_vision
    else:
        npy_files = sorted(glob(os.path.join(model_dir, "**", "*.npy"),
                                 recursive=True))
        extractor = extract_layer_pairs_audio

    if len(npy_files) == 0:
        return None

    all_layers_data = defaultdict(list)
    n_layers = None

    for npy_file in npy_files:
        try:
            data = np.load(npy_file).astype(np.float32)
        except Exception:
            continue

        if data.ndim != 3:
            continue

        valid = extractor(data)
        if valid is None:
            continue

        if n_layers is None:
            n_layers = valid.shape[0]
        elif valid.shape[0] != n_layers:
            continue

        for li in range(n_layers):
            all_layers_data[li].append(valid[li])

    if n_layers is None or n_layers < 2:
        return None

    layer_arrays = {}
    for li in range(n_layers):
        if li not in all_layers_data or len(all_layers_data[li]) == 0:
            return None
        layer_arrays[li] = np.concatenate(all_layers_data[li], axis=0)

    N = layer_arrays[0].shape[0]

    results = {}
    for k in range(n_layers - 1):
        X = torch.from_numpy(layer_arrays[k]).float().to(DEVICE)
        Y = torch.from_numpy(layer_arrays[k + 1]).float().to(DEVICE)

        r2, rel_err = ridge_fit_gpu(X, Y, alpha=alpha)

        r2_shufs, err_shufs = [], []
        for _ in range(n_shuffle):
            perm = torch.randperm(N, device=DEVICE)
            Y_shuf = Y[perm]
            r2_s, err_s = ridge_fit_gpu(X, Y_shuf, alpha=alpha)
            r2_shufs.append(r2_s)
            err_shufs.append(err_s)

        results[k] = {
            'r2': r2,
            'rel_err': rel_err,
            'r2_shuffle': float(np.mean(r2_shufs)),
            'rel_err_shuffle': float(np.mean(err_shufs)),
            'n_stimuli': N
        }

        del X, Y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def collect_modality(data_root, alpha=1.0, n_shuffle=10, fmt="vision"):
    """
    Run layer-wise validation for all models under one modality.
    """
    model_dirs = sorted([
        d for d in glob(os.path.join(data_root, "*"))
        if os.path.isdir(d)
    ])
    print(f"Found {len(model_dirs)} models in {data_root}\n")

    all_r2, all_rel_err = [], []
    all_r2_shuf, all_rel_err_shuf = [], []
    model_results = {}

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"  {model_name} ...", end=" ", flush=True)

        results = validate_model_layerwise(model_dir, alpha=alpha,
                                            n_shuffle=n_shuffle, fmt=fmt)
        if results is None:
            print("skipped")
            continue

        r2s = [v['r2'] for v in results.values()]
        errs = [v['rel_err'] for v in results.values()]
        r2s_s = [v['r2_shuffle'] for v in results.values()]
        errs_s = [v['rel_err_shuffle'] for v in results.values()]
        n = results[0]['n_stimuli']

        all_r2.extend(r2s)
        all_rel_err.extend(errs)
        all_r2_shuf.extend(r2s_s)
        all_rel_err_shuf.extend(errs_s)
        model_results[model_name] = results

        print(f"N={n}, {len(r2s)} layers, "
              f"R²={np.mean(r2s):.3f} (shuf={np.mean(r2s_s):.3f}), "
              f"E_rel={np.mean(errs):.3f} (shuf={np.mean(errs_s):.3f})")

    return (np.array(all_r2), np.array(all_rel_err),
            np.array(all_r2_shuf), np.array(all_rel_err_shuf),
            model_results)


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def main():
    modalities = {
        "Vision":   ("filterData/img/design_matrix",   "vision"),
        "Audio":    ("filterData/audio/design_matrix", "audio"),
        "Language": ("filterData/lang_new/design_matrix",  "audio"),  # same format
    }

    alpha = 1.0
    n_shuffle = 10

    summary = {}

    for modality, (data_root, fmt) in modalities.items():
        if not os.path.exists(data_root):
            print(f"Skipping {modality}: {data_root} not found")
            summary[modality] = None
            continue

        print(f"\n{'='*60}")
        print(f"  Modality: {modality}")
        print(f"{'='*60}")

        all_r2, all_rel_err, all_r2_shuf, all_rel_err_shuf, model_results = \
            collect_modality(data_root, alpha=alpha,
                             n_shuffle=n_shuffle, fmt=fmt)

        if len(all_r2) == 0:
            summary[modality] = None
            continue

        n = len(all_r2)
        _, r2_p = stats.wilcoxon(all_r2, all_r2_shuf, alternative='greater')
        _, err_p = stats.wilcoxon(all_rel_err, all_rel_err_shuf,
                                   alternative='less')

        k_r2 = int(np.sum(all_r2 >= 0.90))
        ci_r2 = clopper_pearson(k_r2, n)

        k_err = int(np.sum(all_rel_err < 0.10))
        ci_err = clopper_pearson(k_err, n)

        summary[modality] = {
            'r2_mean':       np.mean(all_r2),
            'r2_std':        np.std(all_r2),
            'delta_r2':      np.mean(all_r2) - np.mean(all_r2_shuf),
            'r2_above_90':   k_r2 / n,
            'r2_ci':         ci_r2,
            'err_mean':      np.mean(all_rel_err),
            'err_std':       np.std(all_rel_err),
            'err_below_10':  k_err / n,
            'err_ci':        ci_err,
            'p_value':       min(r2_p, err_p),
        }

        # Save arrays
        save_path = f"filterData/layerwise_validation_{modality.lower()}.npz"
        np.savez(save_path,
                 all_r2=all_r2, all_rel_err=all_rel_err,
                 all_r2_shuf=all_r2_shuf, all_rel_err_shuf=all_rel_err_shuf,
                 model_names=list(model_results.keys()))

    # Final cross-modality table
    print(f"\n\n{'='*130}")
    print(f"  CROSS-MODALITY SUMMARY")
    print(f"{'='*130}")
    header = (f"{'Modality':<10} {'R² (mean±std)':<18} {'ΔR² vs shuf':<14} "
              f"{'R² ≥ 0.90':<22} {'E_rel (mean±std)':<20} "
              f"{'E_rel < 0.10':<22} {'p-value':<12}")
    print(header)
    print("-" * 130)

    for modality in ["Vision", "Audio", "Language"]:
        s = summary.get(modality)
        if s is None:
            row = (f"{modality:<10} {'—':<18} {'—':<14} {'—':<22} "
                   f"{'—':<20} {'—':<22} {'—':<12}")
        else:
            r2_str    = f"{s['r2_mean']:.3f} ± {s['r2_std']:.3f}"
            dr2_str   = f"↑ {s['delta_r2']:.3f}"
            r2_90_str = (f"{s['r2_above_90']*100:.1f}% "
                         f"[{s['r2_ci'][0]*100:.1f}, {s['r2_ci'][1]*100:.1f}]")
            err_str   = f"{s['err_mean']:.3f} ± {s['err_std']:.3f}"
            err_10_str = (f"{s['err_below_10']*100:.1f}% "
                          f"[{s['err_ci'][0]*100:.1f}, {s['err_ci'][1]*100:.1f}]")
            p_str     = f"{s['p_value']:.1e}"
            row = (f"{modality:<10} {r2_str:<18} {dr2_str:<14} "
                   f"{r2_90_str:<22} {err_str:<20} {err_10_str:<22} {p_str:<12}")
        print(row)

    print("=" * 130)


if __name__ == "__main__":
    main()