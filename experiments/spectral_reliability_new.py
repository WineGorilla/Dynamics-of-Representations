"""
Spectral Reliability Diagnostics for TIMs (新数据集版)
=====================================================
适配 embeddings/ 目录结构:
  embeddings/vision/{model}.npy          单文件 (n_layers, n_images, feat_dim)
  embeddings/audio/{model}/*.npy         每条音频一个文件 (n_layers, n_chunks, feat_dim)
  embeddings/language/{model}/*.npy      每个文本一个文件 (n_layers, n_bins, feat_dim)

Three diagnostics:
  1. Perturbation Stability
  2. Rank Sensitivity
  3. Bootstrap Resampling

Usage:
  python experiments/spectral_reliability_new.py --modality all --max_traj 500
  python experiments/spectral_reliability_new.py --modality vision --max_traj 500
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


# =============================================================================
# 1. Core DMD
# =============================================================================

def choose_rank(S, threshold=0.9):
    total = np.sum(S ** 2)
    if total < 1e-16:
        return 1
    energy = np.cumsum(S ** 2) / total
    r = np.searchsorted(energy, threshold) + 1
    return int(r)


def compute_dmd_eigenvalues(X, r=None, energy_threshold=0.9, eps=1e-8, center=True):
    X = np.asarray(X, dtype=np.float32)
    L, D = X.shape
    if L < 3:
        return None

    X1 = X[:-1].T
    X2 = X[1:].T

    if center:
        mu = X1.mean(axis=1, keepdims=True)
        X1 = X1 - mu
        X2 = X2 - mu

    U, S, Vt = np.linalg.svd(X1, full_matrices=False)

    if r is None:
        r = choose_rank(S, energy_threshold)
    r = min(r, S.size)
    if r < 1:
        return None

    U = U[:, :r]
    S = S[:r]
    V = Vt[:r, :].T
    invS = 1.0 / (S + eps)

    A = U.T @ ((X2 @ V) * invS)
    eigvals, _ = np.linalg.eig(A)

    return eigvals


# =============================================================================
# 2. Check 1: Perturbation Stability
# =============================================================================

def perturbation_stability(X, noise_levels=(0.001, 0.01, 0.05),
                           n_repeats=10, energy_threshold=0.9, seed=42):
    from scipy.stats import ks_2samp

    X = np.asarray(X, dtype=np.float32)

    clean_eigs = compute_dmd_eigenvalues(X, energy_threshold=energy_threshold)
    if clean_eigs is None:
        return None

    clean_abs = np.sort(np.abs(clean_eigs))
    clean_mean = np.mean(clean_abs)

    rng = np.random.RandomState(seed)
    x_norm = np.linalg.norm(X, 'fro')
    if x_norm < 1e-12:
        return None

    results = {}
    for noise_level in noise_levels:
        shifts = []
        ks_stats = []
        all_noisy = []

        for _ in range(n_repeats):
            noise = rng.randn(*X.shape).astype(np.float32) * noise_level * x_norm / np.sqrt(X.size)
            X_noisy = X + noise

            noisy_eigs = compute_dmd_eigenvalues(X_noisy, energy_threshold=energy_threshold)
            if noisy_eigs is None:
                continue

            noisy_abs = np.sort(np.abs(noisy_eigs))
            all_noisy.append(noisy_abs)
            shifts.append(abs(np.mean(noisy_abs) - clean_mean))
            stat, _ = ks_2samp(clean_abs, noisy_abs)
            ks_stats.append(stat)

        results[noise_level] = {
            'mean_shift': float(np.mean(shifts)) if shifts else float('nan'),
            'std_shift': float(np.std(shifts)) if shifts else float('nan'),
            'ks_stat': float(np.mean(ks_stats)) if ks_stats else float('nan'),
            'ks_std': float(np.std(ks_stats)) if ks_stats else float('nan'),
            'n_success': len(shifts),
        }

    return {'clean_mean': clean_mean, 'clean_eigs': clean_abs, 'noise_results': results}


# =============================================================================
# 3. Check 2: Rank Sensitivity
# =============================================================================

def rank_sensitivity_model(X, energy_thresholds=(0.80, 0.85, 0.90, 0.95, 0.99)):
    X = np.asarray(X, dtype=np.float32)
    L, D = X.shape
    if L < 3:
        return None

    X1 = X[:-1].T
    X2 = X[1:].T
    mu = X1.mean(axis=1, keepdims=True)
    X1c = X1 - mu
    X2c = X2 - mu

    _, S_full, _ = np.linalg.svd(X1c, full_matrices=False)

    results = {}
    for th in energy_thresholds:
        r = choose_rank(S_full, th)
        r = min(r, S_full.size)
        if r < 1:
            continue

        eigs = compute_dmd_eigenvalues(X, r=r)
        if eigs is not None:
            abs_eigs = np.abs(eigs)
            results[th] = {
                'mean_abs_lambda': float(np.mean(abs_eigs)),
                'median_abs_lambda': float(np.median(abs_eigs)),
                'rank': r,
                'eigvals': abs_eigs,
            }

    return results


# =============================================================================
# 4. Check 3: Bootstrap Layer Resampling
# =============================================================================

def bootstrap_layers(X, n_bootstrap=50, frac=0.8, energy_threshold=0.9, seed=42):
    X = np.asarray(X, dtype=np.float32)
    L, D = X.shape
    if L < 4:
        return None

    full_eigs = compute_dmd_eigenvalues(X, energy_threshold=energy_threshold)
    if full_eigs is None:
        return None
    full_mean = float(np.mean(np.abs(full_eigs)))

    rng = np.random.RandomState(seed)
    n_sample = max(3, int(L * frac))

    bootstrap_means = []
    for _ in range(n_bootstrap):
        idx = np.sort(rng.choice(L, size=n_sample, replace=True))
        X_sub = X[idx]

        eigs = compute_dmd_eigenvalues(X_sub, energy_threshold=energy_threshold)
        if eigs is not None:
            bootstrap_means.append(float(np.mean(np.abs(eigs))))

    if len(bootstrap_means) < 5:
        return None

    bootstrap_means = np.array(bootstrap_means)

    return {
        'full_mean': full_mean,
        'bootstrap_means': bootstrap_means,
        'bootstrap_mean_of_means': float(np.mean(bootstrap_means)),
        'bootstrap_std': float(np.std(bootstrap_means)),
        'ci_low': float(np.percentile(bootstrap_means, 2.5)),
        'ci_high': float(np.percentile(bootstrap_means, 97.5)),
    }


# =============================================================================
# 5. File I/O (适配 embeddings/ 新目录结构)
# =============================================================================

def iter_trajectories(model, root, modality, max_traj=None, seed=42):
    """
    Yield (L, d) trajectory matrices.
    - vision:   embeddings/vision/{model}.npy  → 单文件 (L, N, d), 遍历 N
    - audio:    embeddings/audio/{model}/*.npy → 每个文件 (L, T, d), 遍历 T
    - language: embeddings/language/{model}/*.npy → 每个文件 (L, T, d), 遍历 T
    """
    count = 0

    if modality == "vision":
        # 视觉：单个 .npy 文件
        npy_path = os.path.join(root, f"{model}.npy")
        if not os.path.exists(npy_path):
            return
        X = np.load(npy_path)  # (L, N, d)
        if X.ndim != 3:
            return
        L, N, d = X.shape
        for t in range(N):
            yield X[:, t, :]
            count += 1
            if max_traj and count >= max_traj:
                return
    else:
        # 音频/语言：每个文件分别遍历
        model_dir = os.path.join(root, model)
        npy_files = sorted(glob(os.path.join(model_dir, "*.npy")))

        for in_path in npy_files:
            X = np.load(in_path)  # (L, T, d)
            if X.ndim != 3:
                continue
            L, T, d = X.shape
            for t in range(T):
                yield X[:, t, :]
                count += 1
                if max_traj and count >= max_traj:
                    return


# =============================================================================
# 6. Per-Model Collection
# =============================================================================

def run_perturbation_check(model, root, modality, max_traj=500,
                           noise_levels=(0.001, 0.01, 0.05), n_repeats=10):
    print(f"  [Perturbation] {model}")

    all_clean_means = []
    noise_agg = {nl: {'shifts': [], 'ks_stats': []} for nl in noise_levels}

    count = 0
    for traj in iter_trajectories(model, root, modality, max_traj=max_traj):
        result = perturbation_stability(traj, noise_levels=noise_levels,
                                        n_repeats=n_repeats, seed=42 + count)
        if result is None:
            continue

        all_clean_means.append(result['clean_mean'])
        for nl in noise_levels:
            nr = result['noise_results'][nl]
            if not np.isnan(nr['mean_shift']):
                noise_agg[nl]['shifts'].append(nr['mean_shift'])
                noise_agg[nl]['ks_stats'].append(nr['ks_stat'])

        count += 1
        if count % 200 == 0:
            print(f"    processed {count} trajectories...")

    if not all_clean_means:
        return None

    summary = {
        'model': model,
        'n_trajectories': count,
        'clean_mean_lambda': float(np.mean(all_clean_means)),
    }

    for nl in noise_levels:
        shifts = noise_agg[nl]['shifts']
        ks = noise_agg[nl]['ks_stats']
        summary[f'noise_{nl}_mean_shift'] = float(np.mean(shifts)) if shifts else float('nan')
        summary[f'noise_{nl}_std_shift'] = float(np.std(shifts)) if shifts else float('nan')
        summary[f'noise_{nl}_ks_stat'] = float(np.mean(ks)) if ks else float('nan')
        summary[f'noise_{nl}_relative_shift'] = (
            float(np.mean(shifts) / (abs(np.mean(all_clean_means)) + 1e-12))
            if shifts else float('nan')
        )

    return summary


def run_rank_check(model, root, modality, max_traj=500,
                   energy_thresholds=(0.80, 0.85, 0.90, 0.95, 0.99)):
    print(f"  [Rank Sensitivity] {model}")

    th_means = {th: [] for th in energy_thresholds}
    th_ranks = {th: [] for th in energy_thresholds}

    count = 0
    for traj in iter_trajectories(model, root, modality, max_traj=max_traj):
        result = rank_sensitivity_model(traj, energy_thresholds=energy_thresholds)
        if result is None:
            continue

        for th in energy_thresholds:
            if th in result:
                th_means[th].append(result[th]['mean_abs_lambda'])
                th_ranks[th].append(result[th]['rank'])

        count += 1

    if count == 0:
        return None

    summary = {'model': model, 'n_trajectories': count}
    for th in energy_thresholds:
        vals = th_means[th]
        ranks = th_ranks[th]
        summary[f'th_{th}_mean_lambda'] = float(np.mean(vals)) if vals else float('nan')
        summary[f'th_{th}_std_lambda'] = float(np.std(vals)) if vals else float('nan')
        summary[f'th_{th}_mean_rank'] = float(np.mean(ranks)) if ranks else float('nan')

    return summary


def run_bootstrap_check(model, root, modality, max_traj=500,
                        n_bootstrap=50):
    print(f"  [Bootstrap] {model}")

    full_means = []
    boot_means = []
    boot_stds = []
    ci_lows = []
    ci_highs = []

    count = 0
    for traj in iter_trajectories(model, root, modality, max_traj=max_traj):
        result = bootstrap_layers(traj, n_bootstrap=n_bootstrap, seed=42 + count)
        if result is None:
            continue

        full_means.append(result['full_mean'])
        boot_means.append(result['bootstrap_mean_of_means'])
        boot_stds.append(result['bootstrap_std'])
        ci_lows.append(result['ci_low'])
        ci_highs.append(result['ci_high'])

        count += 1

    if count == 0:
        return None

    ci_widths = np.array(ci_highs) - np.array(ci_lows)
    bias = np.array(boot_means) - np.array(full_means)

    return {
        'model': model,
        'n_trajectories': count,
        'full_mean_lambda': float(np.mean(full_means)),
        'bootstrap_mean_lambda': float(np.mean(boot_means)),
        'mean_bias': float(np.mean(bias)),
        'mean_ci_width': float(np.mean(ci_widths)),
        'median_ci_width': float(np.median(ci_widths)),
        'mean_bootstrap_std': float(np.mean(boot_stds)),
        'frac_full_in_ci': float(np.mean(
            (np.array(full_means) >= np.array(ci_lows)) &
            (np.array(full_means) <= np.array(ci_highs))
        )),
    }


# =============================================================================
# 7. Visualization
# =============================================================================

def plot_perturbation_summary(results_by_modality, noise_levels, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    modalities = list(results_by_modality.keys())
    n_mod = len(modalities)
    n_noise = len(noise_levels)
    width = 0.8 / n_mod
    colors = {"vision": "#e74c3c", "audio": "#3498db", "language": "#2ecc71"}

    for i, mod in enumerate(modalities):
        models_data = results_by_modality[mod]
        rel_shifts = []
        for nl in noise_levels:
            vals = [m[f'noise_{nl}_relative_shift'] for m in models_data
                    if not np.isnan(m.get(f'noise_{nl}_relative_shift', float('nan')))]
            rel_shifts.append(np.mean(vals) if vals else 0)

        x = np.arange(n_noise) + i * width
        ax.bar(x, rel_shifts, width, label=mod.capitalize(),
               color=colors.get(mod, '#999'), alpha=0.85, edgecolor='white')

    ax.set_xticks(np.arange(n_noise) + width * (n_mod - 1) / 2)
    ax.set_xticklabels([f"ε={nl}" for nl in noise_levels])
    ax.set_ylabel("Relative Shift in Mean |λ|", fontsize=12)
    ax.set_title("Eigenvalue Perturbation Stability", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_rank_sensitivity(results_by_modality, energy_thresholds, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"vision": "#e74c3c", "audio": "#3498db", "language": "#2ecc71"}

    for mod, models_data in results_by_modality.items():
        means = []
        stds = []
        for th in energy_thresholds:
            vals = [m[f'th_{th}_mean_lambda'] for m in models_data
                    if not np.isnan(m.get(f'th_{th}_mean_lambda', float('nan')))]
            means.append(np.mean(vals) if vals else float('nan'))
            stds.append(np.std(vals) if vals else 0)

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(energy_thresholds, means, 'o-', color=colors.get(mod, '#999'),
                label=mod.capitalize(), lw=2, markersize=6)
        ax.fill_between(energy_thresholds, means - stds, means + stds,
                        color=colors.get(mod, '#999'), alpha=0.15)

    ax.set_xlabel("SVD Energy Threshold", fontsize=12)
    ax.set_ylabel("Mean |λ| (model-averaged)", fontsize=12)
    ax.set_title("Rank Sensitivity: Modality Separation Across Thresholds", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_rank_with_rank_axis(results_by_modality, energy_thresholds, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"vision": "#e74c3c", "audio": "#3498db", "language": "#2ecc71"}

    for mod, models_data in results_by_modality.items():
        mean_lambdas = []
        mean_ranks = []
        for th in energy_thresholds:
            lam_vals = [m[f'th_{th}_mean_lambda'] for m in models_data
                        if not np.isnan(m.get(f'th_{th}_mean_lambda', float('nan')))]
            rank_vals = [m[f'th_{th}_mean_rank'] for m in models_data
                         if not np.isnan(m.get(f'th_{th}_mean_rank', float('nan')))]
            mean_lambdas.append(np.mean(lam_vals) if lam_vals else float('nan'))
            mean_ranks.append(np.mean(rank_vals) if rank_vals else float('nan'))

        ax.plot(mean_ranks, mean_lambdas, 'o-', color=colors.get(mod, '#999'),
                label=mod.capitalize(), lw=2, markersize=6)

    ax.set_xlabel("Mean Rank r", fontsize=12)
    ax.set_ylabel("Mean |λ|", fontsize=12)
    ax.set_title("Mean |λ| vs. Rank Across Modalities", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_bootstrap_summary(results_by_modality, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"vision": "#e74c3c", "audio": "#3498db", "language": "#2ecc71"}

    modalities = list(results_by_modality.keys())

    ax = axes[0]
    ci_data = []
    labels = []
    for mod in modalities:
        vals = [m['mean_ci_width'] for m in results_by_modality[mod]]
        ci_data.append(vals)
        labels.append(mod.capitalize())

    bp = ax.boxplot(ci_data, labels=labels, patch_artist=True, widths=0.5)
    for patch, mod in zip(bp['boxes'], modalities):
        patch.set_facecolor(colors.get(mod, '#999'))
        patch.set_alpha(0.7)
    ax.set_ylabel("95% CI Width of Mean |λ|", fontsize=12)
    ax.set_title("Bootstrap Stability", fontsize=13)

    ax = axes[1]
    coverage = []
    for mod in modalities:
        vals = [m['frac_full_in_ci'] for m in results_by_modality[mod]]
        coverage.append(vals)

    bp = ax.boxplot(coverage, labels=labels, patch_artist=True, widths=0.5)
    for patch, mod in zip(bp['boxes'], modalities):
        patch.set_facecolor(colors.get(mod, '#999'))
        patch.set_alpha(0.7)
    ax.set_ylabel("Coverage (full mean in 95% CI)", fontsize=12)
    ax.set_title("Bootstrap Coverage", fontsize=13)
    ax.axhline(0.95, color='gray', ls='--', lw=1, label='nominal 95%')
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# 8. Print Tables
# =============================================================================

def print_perturbation_table(results_by_modality, noise_levels):
    print("\n" + "=" * 100)
    print("CHECK 1: PERTURBATION STABILITY")
    print("=" * 100)

    for mod, models_data in results_by_modality.items():
        print(f"\n  {mod.upper()} ({len(models_data)} models)")
        nl_headers = "  ".join([f"ε={nl}: shift(rel)" for nl in noise_levels])
        print(f"  {'Model':<35} | {nl_headers}")
        print(f"  {'-' * 90}")

        for m in models_data:
            vals = []
            for nl in noise_levels:
                rs = m.get(f'noise_{nl}_relative_shift', float('nan'))
                ks = m.get(f'noise_{nl}_ks_stat', float('nan'))
                vals.append(f"{rs:.4f} (KS={ks:.3f})")
            print(f"  {m['model']:<35} | {'  '.join(vals)}")

    print(f"\n  {'SUMMARY':<35}", end="")
    for nl in noise_levels:
        print(f" | ε={nl}", end="")
    print()
    print(f"  {'-' * 90}")
    for mod, models_data in results_by_modality.items():
        print(f"  {mod:<35}", end="")
        for nl in noise_levels:
            vals = [m[f'noise_{nl}_relative_shift'] for m in models_data
                    if not np.isnan(m.get(f'noise_{nl}_relative_shift', float('nan')))]
            mean_rs = np.mean(vals) if vals else float('nan')
            print(f" | {mean_rs:.4f}", end="")
        print()

    print("=" * 100)


def print_rank_table(results_by_modality, energy_thresholds):
    print("\n" + "=" * 100)
    print("CHECK 2: RANK SENSITIVITY")
    print("=" * 100)

    th_headers = "  ".join([f"E={th}" for th in energy_thresholds])
    print(f"\n  {'Modality':<12} | {th_headers}")
    print(f"  {'-' * 80}")

    for mod, models_data in results_by_modality.items():
        vals = []
        for th in energy_thresholds:
            model_means = [m[f'th_{th}_mean_lambda'] for m in models_data
                           if not np.isnan(m.get(f'th_{th}_mean_lambda', float('nan')))]
            mean_val = np.mean(model_means) if model_means else float('nan')
            std_val = np.std(model_means) if model_means else float('nan')
            vals.append(f"{mean_val:.3f}±{std_val:.3f}")
        print(f"  {mod:<12} | {'  '.join(vals)}")

    print(f"\n  Mean rank r:")
    print(f"  {'Modality':<12} | {th_headers}")
    print(f"  {'-' * 80}")
    for mod, models_data in results_by_modality.items():
        vals = []
        for th in energy_thresholds:
            rank_means = [m[f'th_{th}_mean_rank'] for m in models_data
                          if not np.isnan(m.get(f'th_{th}_mean_rank', float('nan')))]
            mean_r = np.mean(rank_means) if rank_means else float('nan')
            vals.append(f"{mean_r:.1f}")
        print(f"  {mod:<12} | {'       '.join(vals)}")

    print("=" * 100)


def print_bootstrap_table(results_by_modality):
    print("\n" + "=" * 100)
    print("CHECK 3: BOOTSTRAP LAYER RESAMPLING")
    print("=" * 100)

    print(f"\n  {'Modality':<12} | {'Mean|λ|':>8} | {'Boot Mean':>9} | {'Bias':>8} | "
          f"{'CI Width':>8} | {'Boot Std':>8} | {'Coverage':>8}")
    print(f"  {'-' * 80}")

    for mod, models_data in results_by_modality.items():
        full = np.mean([m['full_mean_lambda'] for m in models_data])
        boot = np.mean([m['bootstrap_mean_lambda'] for m in models_data])
        bias = np.mean([m['mean_bias'] for m in models_data])
        ci_w = np.mean([m['mean_ci_width'] for m in models_data])
        bstd = np.mean([m['mean_bootstrap_std'] for m in models_data])
        cov = np.mean([m['frac_full_in_ci'] for m in models_data])

        print(f"  {mod:<12} | {full:>8.4f} | {boot:>9.4f} | {bias:>+8.4f} | "
              f"{ci_w:>8.4f} | {bstd:>8.4f} | {cov:>8.1%}")

    print("=" * 100)


def print_latex_tables(pert_results, rank_results, boot_results, noise_levels, energy_thresholds):
    print("\n% ===== LaTeX: Perturbation Stability =====")
    print(r"\begin{tabular}{l" + " c" * len(noise_levels) + "}")
    print(r"\toprule")
    print("Modality & " + " & ".join([f"$\\epsilon={nl}$" for nl in noise_levels]) + r" \\")
    print(r"\midrule")
    for mod, models_data in pert_results.items():
        vals = []
        for nl in noise_levels:
            rs = [m[f'noise_{nl}_relative_shift'] for m in models_data
                  if not np.isnan(m.get(f'noise_{nl}_relative_shift', float('nan')))]
            vals.append(f"${np.mean(rs):.4f}$" if rs else "--")
        print(f"{mod.capitalize()} & " + " & ".join(vals) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\n% ===== LaTeX: Rank Sensitivity =====")
    print(r"\begin{tabular}{l" + " c" * len(energy_thresholds) + "}")
    print(r"\toprule")
    print("Modality & " + " & ".join([f"$E={th}$" for th in energy_thresholds]) + r" \\")
    print(r"\midrule")
    for mod, models_data in rank_results.items():
        vals = []
        for th in energy_thresholds:
            lam = [m[f'th_{th}_mean_lambda'] for m in models_data
                   if not np.isnan(m.get(f'th_{th}_mean_lambda', float('nan')))]
            vals.append(f"${np.mean(lam):.3f} \\pm {np.std(lam):.3f}$" if lam else "--")
        print(f"{mod.capitalize()} & " + " & ".join(vals) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\n% ===== LaTeX: Bootstrap =====")
    print(r"\begin{tabular}{l c c c c c}")
    print(r"\toprule")
    print(r"Modality & Mean $|\lambda|$ & Bias & CI Width & Std & Coverage \\")
    print(r"\midrule")
    for mod, models_data in boot_results.items():
        full = np.mean([m['full_mean_lambda'] for m in models_data])
        bias = np.mean([m['mean_bias'] for m in models_data])
        ci_w = np.mean([m['mean_ci_width'] for m in models_data])
        bstd = np.mean([m['mean_bootstrap_std'] for m in models_data])
        cov = np.mean([m['frac_full_in_ci'] for m in models_data])
        print(f"{mod.capitalize()} & ${full:.4f}$ & ${bias:+.4f}$ & "
              f"${ci_w:.4f}$ & ${bstd:.4f}$ & ${cov:.1%}$ " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# =============================================================================
# 9. Model Lists & Config (新数据集)
# =============================================================================

VISION_MODELS = [
    "beit-base-patch16-224-pt22k-ft22k",
    "beit-large-patch16-224-pt22k-ft22k",
    "clip-vit-base-patch16",
    "clip-vit-base-patch32",
    "clip-vit-large-patch14",
    "convnext_base",
    "convnext_tiny",
    "data2vec-vision-base",
    "data2vec-vision-large",
    "deit-base-patch16-224",
    "deit-small-patch16-224",
    "densenet121",
    "densenet201",
    "dino-vitb16",
    "dino-vits16",
    "dinov2-base",
    "dinov2-large",
    "dinov2-small",
    "efficientnet_b0",
    "efficientnet_b4",
    "resnet50",
    "resnet101",
    "sam-vit-base",
    "sam-vit-large",
    "sam-vit-huge",
    "swin-tiny-patch4-window7-224",
    "swin-small-patch4-window7-224",
    "swin-large-patch4-window7-224",
    "vgg16",
    "vgg19",
    "vit-base-patch16-224-in21k",
    "vit-large-patch16-224-in21k",
    "vit-mae-base",
    "vit-mae-large",
    "vit-msn-base",
    "vit-msn-large",
]

AUDIO_MODELS = [
    "data2vec-audio-base",
    "data2vec-audio-base-960h",
    "data2vec-audio-large",
    "data2vec-audio-large-960h",
    "hubert-base-ls960",
    "hubert-base-superb-ks",
    "hubert-large-ls960-ft",
    "hubert-xlarge-ls960-ft",
    "sew-d-mid-100k",
    "sew-d-small-100k",
    "sew-d-tiny-100k",
    "sew-mid-100k",
    "sew-small-100k",
    "sew-tiny-100k",
    "unispeech-large-1500h-cv",
    "unispeech-sat-base",
    "unispeech-sat-base-plus",
    "unispeech-sat-large",
    "w2v-bert-2.0",
    "wav2vec2-base",
    "wav2vec2-base-960h",
    "wav2vec2-base-superb-ks",
    "wav2vec2-conformer-rel-pos-large",
    "wav2vec2-conformer-rope-large-960h-ft",
    "wav2vec2-large",
    "wav2vec2-large-960h",
    "wav2vec2-large-xlsr-53",
    "wav2vec2-xls-r-1b",
    "wav2vec2-xls-r-300m",
    "wavlm-base",
    "wavlm-base-plus",
    "wavlm-large",
    "whisper-base",
    "whisper-medium",
    "whisper-small",
    "whisper-tiny",
]

LANGUAGE_MODELS = [
    "MiniLM-L6-H384-uncased",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "albert-base-v2",
    "albert-large-v2",
    "albert-xlarge-v2",
    "bert-base-cased",
    "bert-base-multilingual-cased",
    "bert-base-uncased",
    "bert-large-cased",
    "bert-large-uncased",
    "camembert-base",
    "conv-bert-base",
    "conv-bert-medium-small",
    "data2vec-text-base",
    "deberta-base",
    "deberta-large",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased",
    "distilroberta-base",
    "electra-base-discriminator",
    "electra-large-discriminator",
    "electra-small-discriminator",
    "ernie-2.0-base-en",
    "ernie-2.0-large-en",
    "ibert-roberta-base",
    "mpnet-base",
    "rembert",
    "roberta-base",
    "roberta-large",
    "squeezebert-uncased",
    "t5-small",
    "xlm-roberta-base",
    "xlm-roberta-large",
    "xlnet-base-cased",
    "xlnet-large-cased",
]

MODALITY_CONFIG = {
    "vision":   {"root": "embeddings/vision",    "models": VISION_MODELS},
    "audio":    {"root": "embeddings/audio",      "models": AUDIO_MODELS},
    "language": {"root": "embeddings/language",   "models": LANGUAGE_MODELS},
}


# =============================================================================
# 10. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TIMs Spectral Reliability (新数据集)")
    parser.add_argument("--modality", type=str, default="all",
                        choices=["vision", "audio", "language", "all"])
    parser.add_argument("--out_dir", type=str, default="results/spectral_reliability_new")
    parser.add_argument("--max_traj", type=int, default=500)
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.001, 0.01, 0.05])
    parser.add_argument("--n_noise_repeats", type=int, default=10)
    parser.add_argument("--energy_thresholds", type=float, nargs="+",
                        default=[0.80, 0.85, 0.90, 0.95, 0.99])
    parser.add_argument("--n_bootstrap", type=int, default=50)
    parser.add_argument("--skip_perturbation", action="store_true")
    parser.add_argument("--skip_rank", action="store_true")
    parser.add_argument("--skip_bootstrap", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    modalities = list(MODALITY_CONFIG.keys()) if args.modality == "all" else [args.modality]
    noise_levels = tuple(args.noise_levels)
    energy_thresholds = tuple(args.energy_thresholds)

    pert_results = {}
    rank_results = {}
    boot_results = {}

    for mod in modalities:
        cfg = MODALITY_CONFIG[mod]
        models = cfg["models"]
        root = cfg["root"]

        if not args.skip_perturbation:
            print(f"\n{'='*60}")
            print(f" CHECK 1 - PERTURBATION STABILITY: {mod.upper()}")
            print(f"{'='*60}")

            pert_results[mod] = []
            for model in tqdm(models, desc=f"Perturbation [{mod}]"):
                r = run_perturbation_check(model, root, mod,
                                           max_traj=args.max_traj,
                                           noise_levels=noise_levels,
                                           n_repeats=args.n_noise_repeats)
                if r:
                    pert_results[mod].append(r)

        if not args.skip_rank:
            print(f"\n{'='*60}")
            print(f" CHECK 2 - RANK SENSITIVITY: {mod.upper()}")
            print(f"{'='*60}")

            rank_results[mod] = []
            for model in tqdm(models, desc=f"Rank [{mod}]"):
                r = run_rank_check(model, root, mod,
                                   max_traj=args.max_traj,
                                   energy_thresholds=energy_thresholds)
                if r:
                    rank_results[mod].append(r)

        if not args.skip_bootstrap:
            print(f"\n{'='*60}")
            print(f" CHECK 3 - BOOTSTRAP: {mod.upper()}")
            print(f"{'='*60}")

            boot_results[mod] = []
            for model in tqdm(models, desc=f"Bootstrap [{mod}]"):
                r = run_bootstrap_check(model, root, mod,
                                        max_traj=args.max_traj,
                                        n_bootstrap=args.n_bootstrap)
                if r:
                    boot_results[mod].append(r)

    # ── Print tables ──
    if pert_results:
        print_perturbation_table(pert_results, noise_levels)
    if rank_results:
        print_rank_table(rank_results, energy_thresholds)
    if boot_results:
        print_bootstrap_table(boot_results)

    if pert_results and rank_results and boot_results:
        print_latex_tables(pert_results, rank_results, boot_results,
                           noise_levels, energy_thresholds)

    # ── Plots ──
    if pert_results:
        plot_perturbation_summary(pert_results, noise_levels,
                                  os.path.join(args.out_dir, "perturbation_stability.pdf"))
    if rank_results:
        plot_rank_sensitivity(rank_results, energy_thresholds,
                              os.path.join(args.out_dir, "rank_sensitivity.pdf"))
        plot_rank_with_rank_axis(rank_results, energy_thresholds,
                                 os.path.join(args.out_dir, "rank_vs_lambda.pdf"))
    if boot_results:
        plot_bootstrap_summary(boot_results,
                               os.path.join(args.out_dir, "bootstrap_summary.pdf"))

    # ── Save raw results ──
    import json

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    save_dict = {}
    for check_name, check_data in [("perturbation", pert_results),
                                     ("rank", rank_results),
                                     ("bootstrap", boot_results)]:
        for mod, models_data in check_data.items():
            for m in models_data:
                key = f"{check_name}_{mod}_{m.get('model', 'unknown')}"
                save_dict[key] = {k: make_serializable(v) for k, v in m.items()}

    json_path = os.path.join(args.out_dir, "all_results.json")
    with open(json_path, 'w') as f:
        json.dump(save_dict, f, indent=2, default=str)
    print(f"\n  Raw results saved: {json_path}")

    print(f"\n✓ All checks complete. Results in: {args.out_dir}/")


if __name__ == "__main__":
    main()