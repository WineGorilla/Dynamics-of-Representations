"""
Real vs Random Baseline 对比分析
=================================
读取 neweigvals/ 下的真实数据和随机数据，
用 KDE Mode 做模态内/模态间检验，对比分布差异。

输入:
  neweigvals/vision/*.npy          (真实)
  neweigvals/audio/*.npy
  neweigvals/language/*.npy
  neweigvals/random_vision/*.npy   (随机)
  neweigvals/random_audio/*.npy
  neweigvals/random_language/*.npy

用法:
  python experiments/compare_real_vs_random.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import os
from glob import glob
from scipy.stats import gaussian_kde, wilcoxon, kruskal, mannwhitneyu


def compute_kde_mode(eigvals_abs, n_points=1000):
    if len(eigvals_abs) < 3:
        return np.median(eigvals_abs)
    try:
        kde = gaussian_kde(eigvals_abs, bw_method='scott')
        x_grid = np.linspace(eigvals_abs.min(), eigvals_abs.max(), n_points)
        density = kde(x_grid)
        return x_grid[np.argmax(density)]
    except Exception:
        return np.median(eigvals_abs)


def compute_ratio(eigvals_abs, low, high):
    return np.mean((eigvals_abs >= low) & (eigvals_abs <= high)) * 100


def analyze_dir(eigvals_dir, label):
    """分析一个目录下所有模型的 eigvals，返回 per-model 统计"""
    npy_files = sorted(glob(os.path.join(eigvals_dir, "*.npy")))
    if not npy_files:
        print(f"  ⚠️ {label}: 没有找到 .npy 文件")
        return None

    results = []
    for f in npy_files:
        model_name = os.path.basename(f).replace(".npy", "")
        eigvals = np.load(f, allow_pickle=True).flatten()
        eigvals_abs = np.abs(eigvals)

        if np.any(~np.isfinite(eigvals_abs)):
            continue
        if len(eigvals_abs) == 0:
            continue

        mode            = compute_kde_mode(eigvals_abs)
        mean_abs_lambda = float(np.mean(eigvals_abs))
        zero_ratio      = compute_ratio(eigvals_abs, 0.0, 0.1)
        unit_ratio      = compute_ratio(eigvals_abs, 0.9, 1.1)
        growth_ratio    = compute_ratio(eigvals_abs, 1.2, np.inf)

        results.append({
            "model":           model_name,
            "mode":            mode,
            "mean_abs_lambda": mean_abs_lambda,
            "zero_ratio":      zero_ratio,
            "unit_ratio":      unit_ratio,
            "growth_ratio":    growth_ratio,
            "n_eigvals":       len(eigvals_abs),
        })

    if not results:
        return None

    return {
        "label":    label,
        "n_models": len(results),
        "modes":    np.array([r["mode"] for r in results]),
        "means":    np.array([r["mean_abs_lambda"] for r in results]),
        "zeros":    np.array([r["zero_ratio"] for r in results]),
        "units":    np.array([r["unit_ratio"] for r in results]),
        "growths":  np.array([r["growth_ratio"] for r in results]),
        "per_model": results,
    }


def format_p(p):
    if p < 0.0001: return "< 0.0001"
    if p < 0.001:  return "< 0.001"
    if p < 0.01:   return "< 0.01"
    if p < 0.05:   return "< 0.05"
    return f"{p:.4f}"


def sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def print_comparison_table(real, rand, modality_name):
    """打印单个模态的 real vs random 对比"""
    print(f"\n  {'Metric':<20} | {'Real':>20} | {'Random':>20} | {'U-stat':>10} | {'p-value':>12} | {'Sig':>5}")
    print(f"  {'-'*95}")

    metrics = [
        ("KDE Mode",    real["modes"],   rand["modes"]),
        ("Mean |λ|",    real["means"],   rand["means"]),
        ("Zero %",      real["zeros"],   rand["zeros"]),
        ("Unit %",      real["units"],   rand["units"]),
        ("Growth %",    real["growths"], rand["growths"]),
    ]

    for name, r_vals, d_vals in metrics:
        u_stat, p_val = mannwhitneyu(r_vals, d_vals, alternative='two-sided')
        r_str = f"{np.mean(r_vals):.4f} ± {np.std(r_vals):.4f}"
        d_str = f"{np.mean(d_vals):.4f} ± {np.std(d_vals):.4f}"
        print(f"  {name:<20} | {r_str:>20} | {d_str:>20} | {u_stat:>10.1f} | {format_p(p_val):>12} | {sig_star(p_val):>5}")


def main():
    base_dir = "neweigvals"

    modalities = {
        "Vision":   ("vision",   "fmrivision"),
        "Audio":    ("audio",    "fmriaudio"),
        "Language": ("language", "fmrilanguage"),
    }

    all_real = []
    all_rand = []

    for mod_name, (real_sub, rand_sub) in modalities.items():
        real_dir = os.path.join(base_dir, real_sub)
        rand_dir = os.path.join(base_dir, rand_sub)

        print(f"\n{'='*100}")
        print(f"  {mod_name}: Real vs Random")
        print(f"{'='*100}")

        if not os.path.exists(real_dir):
            print(f"  ⚠️ 真实数据不存在: {real_dir}")
            continue
        if not os.path.exists(rand_dir):
            print(f"  ⚠️ 随机数据不存在: {rand_dir}")
            continue

        real = analyze_dir(real_dir, f"{mod_name} (Real)")
        rand = analyze_dir(rand_dir, f"{mod_name} (Random)")

        if real is None or rand is None:
            print(f"  ⚠️ 数据不足，跳过")
            continue

        # 打印 per-model 详情
        print(f"\n  Real ({real['n_models']} models):")
        for r in real["per_model"]:
            print(f"    {r['model']:40s} | mode={r['mode']:.6f} | mean|λ|={r['mean_abs_lambda']:.6f} | "
                  f"zero={r['zero_ratio']:.1f}% | unit={r['unit_ratio']:.1f}% | growth={r['growth_ratio']:.1f}%")

        print(f"\n  Random ({rand['n_models']} models):")
        for r in rand["per_model"]:
            print(f"    {r['model']:40s} | mode={r['mode']:.6f} | mean|λ|={r['mean_abs_lambda']:.6f} | "
                  f"zero={r['zero_ratio']:.1f}% | unit={r['unit_ratio']:.1f}% | growth={r['growth_ratio']:.1f}%")

        # 对比表格
        print_comparison_table(real, rand, mod_name)

        all_real.append((mod_name, real))
        all_rand.append((mod_name, rand))

    # ── 跨模态汇总 ──
    if len(all_real) >= 2:
        print(f"\n\n{'='*100}")
        print(f"  CROSS-MODALITY SUMMARY")
        print(f"{'='*100}")

        print(f"\n  {'Modality':<12} | {'Real Mode':>12} | {'Rand Mode':>12} | "
              f"{'Real Mean|λ|':>14} | {'Rand Mean|λ|':>14} | {'Mode p':>12} | {'Mean|λ| p':>12}")
        print(f"  {'-'*100}")

        for (mod_name, real), (_, rand) in zip(all_real, all_rand):
            _, p_mode = mannwhitneyu(real["modes"], rand["modes"], alternative='two-sided')
            _, p_mean = mannwhitneyu(real["means"], rand["means"], alternative='two-sided')

            print(f"  {mod_name:<12} | {np.mean(real['modes']):>12.4f} | {np.mean(rand['modes']):>12.4f} | "
                  f"{np.mean(real['means']):>14.4f} | {np.mean(rand['means']):>14.4f} | "
                  f"{format_p(p_mode):>12} | {format_p(p_mean):>12}")

        print(f"  {'='*100}")

    # ── 保存 ──
    save_data = {}
    for mod_name, real in all_real:
        k = mod_name.lower()
        save_data[f"{k}_real_modes"] = real["modes"]
        save_data[f"{k}_real_means"] = real["means"]
    for mod_name, rand in all_rand:
        k = mod_name.lower()
        save_data[f"{k}_rand_modes"] = rand["modes"]
        save_data[f"{k}_rand_means"] = rand["means"]

    np.savez("real_vs_random_summary.npz", **save_data)
    print("\n结果已保存到: real_vs_random_summary.npz")


if __name__ == "__main__":
    main()