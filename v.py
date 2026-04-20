"""
Layer-wise Affine Fitting Validation
======================================
For each adjacent layer pair (k, k+1), fit:
  x_{k+1} = A_k @ x_k + c_k
using all stimuli via ridge regression.

Report per-layer R^2 and relative error, aggregate by modality.

Usage:
  CUDA_VISIBLE_DEVICES=2 python v.py
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


def extract_layer_pairs(data):
    """
    From (n_layers, n_tr, d), extract non-zero TRs.
    Returns: (n_layers, N_valid, d) array
    """
    norms = np.linalg.norm(data, axis=2)  # (n_layers, n_tr)
    active_mask = norms.sum(axis=0) > 0
    active_trs = np.where(active_mask)[0]

    # Filter: all layers must be non-zero
    valid = []
    for t in active_trs:
        traj = data[:, t, :]
        if np.any(np.linalg.norm(traj, axis=1) == 0):
            continue
        valid.append(t)

    if len(valid) == 0:
        return None

    return data[:, valid, :]  # (n_layers, N_valid, d)


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


def validate_model_layerwise(model_dir, alpha=1.0):
    """
    Collect all stimuli across runs, then fit each layer pair.
    Returns dict: layer_idx -> (R^2, rel_err, N)
    """
    npy_files = sorted(glob(os.path.join(model_dir, "**",
                                          "*_bold_embedding.npy"), recursive=True))
    if len(npy_files) == 0:
        return None

    # Collect all valid stimuli
    all_layers_data = defaultdict(list)
    n_layers = None

    for npy_file in npy_files:
        try:
            data = np.load(npy_file).astype(np.float32)
        except Exception:
            continue

        valid = extract_layer_pairs(data)
        if valid is None:
            continue

        if n_layers is None:
            n_layers = valid.shape[0]
        elif valid.shape[0] != n_layers:
            continue

        # valid: (n_layers, N, d)
        for li in range(n_layers):
            all_layers_data[li].append(valid[li])

    if n_layers is None or n_layers < 2:
        return None

    # Concatenate across runs: per layer -> (N_total, d)
    layer_arrays = {}
    for li in range(n_layers):
        if li not in all_layers_data or len(all_layers_data[li]) == 0:
            return None
        layer_arrays[li] = np.concatenate(all_layers_data[li], axis=0)

    N = layer_arrays[0].shape[0]

    # Fit each adjacent pair
    results = {}
    for k in range(n_layers - 1):
        X = torch.from_numpy(layer_arrays[k]).float().to(DEVICE)
        Y = torch.from_numpy(layer_arrays[k + 1]).float().to(DEVICE)

        r2, rel_err = ridge_fit_gpu(X, Y, alpha=alpha)
        results[k] = {'r2': r2, 'rel_err': rel_err, 'n_stimuli': N}

        del X, Y
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def collect_modality(data_root, alpha=1.0):
    """
    Run layer-wise validation for all models under one modality.
    """
    model_dirs = sorted([
        d for d in glob(os.path.join(data_root, "*"))
        if os.path.isdir(d)
    ])
    print(f"Found {len(model_dirs)} models in {data_root}\n")

    all_r2 = []
    all_rel_err = []
    model_results = {}

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"  {model_name} ...", end=" ", flush=True)

        results = validate_model_layerwise(model_dir, alpha=alpha)
        if results is None:
            print("skipped")
            continue

        r2s = [v['r2'] for v in results.values()]
        errs = [v['rel_err'] for v in results.values()]
        n = results[0]['n_stimuli']

        all_r2.extend(r2s)
        all_rel_err.extend(errs)
        model_results[model_name] = results

        print(f"N={n}, {len(r2s)} layers, "
              f"R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}, "
              f"E_rel={np.mean(errs):.4f}±{np.std(errs):.4f}")

    return np.array(all_r2), np.array(all_rel_err), model_results


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def main():
    modalities = {
        "Image": "filterData/img/design_matrix",
        # "Language": "filterData/lang/design_matrix",
        # "Audio": "filterData/audio/design_matrix",
    }

    alpha = 1.0  # Ridge regularization
    thresholds_err = [0.05, 0.10]
    thresholds_r2 = [0.90, 0.95]

    for modality, data_root in modalities.items():
        if not os.path.exists(data_root):
            print(f"Skipping {modality}: {data_root} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Modality: {modality}")
        print(f"{'='*60}")

        all_r2, all_rel_err, model_results = collect_modality(
            data_root, alpha=alpha
        )

        if len(all_r2) == 0:
            print("  No valid results")
            continue

        n = len(all_r2)

        print(f"\n\n{'='*70}")
        print(f"  {modality} Modality — Layer-wise Affine Fitting")
        print(f"  N = {n} layer transitions across all models")
        print(f"{'='*70}")

        # R^2 stats
        print(f"\n  R² Statistics:")
        print(f"    mean ± std:     {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
        print(f"    min:            {np.min(all_r2):.4f}")
        print(f"    5th percentile: {np.percentile(all_r2, 5):.4f}")

        for tau in thresholds_r2:
            k = int(np.sum(all_r2 >= tau))
            prop = k / n
            ci_lo, ci_hi = clopper_pearson(k, n)
            stat, pval = stats.binomtest(k, n, p=0.9, alternative='greater').statistic, \
                         stats.binomtest(k, n, p=0.9, alternative='greater').pvalue
            print(f"    R² >= {tau}:       {prop*100:.1f}% "
                  f"(95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%], "
                  f"binomial p={pval:.2e})")

        # Relative error stats
        print(f"\n  E_rel Statistics:")
        print(f"    mean ± std:      {np.mean(all_rel_err):.4f} ± {np.std(all_rel_err):.4f}")
        print(f"    max:             {np.max(all_rel_err):.4f}")
        print(f"    95th percentile: {np.percentile(all_rel_err, 95):.4f}")

        for tau in thresholds_err:
            k = int(np.sum(all_rel_err < tau))
            prop = k / n
            ci_lo, ci_hi = clopper_pearson(k, n)
            pct = int(tau * 100)
            btest = stats.binomtest(k, n, p=0.9, alternative='greater')
            print(f"    E_rel < {tau}:      {prop*100:.1f}% "
                  f"(95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%], "
                  f"binomial p={btest.pvalue:.2e})")

        # Save
        save_path = f"filterData/layerwise_validation_{modality.lower()}.npz"
        np.savez(save_path,
                 all_r2=all_r2,
                 all_rel_err=all_rel_err,
                 model_names=list(model_results.keys()))
        print(f"\n  Saved to {save_path}")


if __name__ == "__main__":
    main()