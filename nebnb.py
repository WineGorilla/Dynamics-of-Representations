"""
模型：KDE 找峰值(mode)，算极点占比(zero-ratio)、单位圆占比(unit-ratio)、增长模态占比(growth-ratio)
- 模态内检验：zero/unit/growth 两两配对 Wilcoxon
- 模态间检验：Kruskal-Wallis + Mann-Whitney U 两两比较
- 输出表格
"""
# CUDA_VISIBLE_DEVICES=1 python nebnb.py

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
    """计算绝对值在 [low, high] 范围内的特征值占比（%）"""
    return np.mean((eigvals_abs >= low) & (eigvals_abs <= high)) * 100


def paired_wilcoxon(a, b):
    """安全的配对 Wilcoxon，返回 (w_stat, p_value, cohens_d)"""
    diff = a - b
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1) if len(diff) > 1 else 1.0
    cohens_d = mean_diff / std_diff if std_diff > 0 else float('inf')

    if len(a) >= 2 and not np.all(diff == 0):
        w_stat, p_value = wilcoxon(a, b, alternative='two-sided')
    else:
        w_stat, p_value = 0.0, 1.0

    return w_stat, p_value, cohens_d, mean_diff


def analyze_modality(modality_dir, modality_name):
    npy_files = sorted(glob(os.path.join(modality_dir, "*.npy")))
    if not npy_files:
        print(f"  ⚠️ {modality_name}: 没有找到 .npy 文件")
        return None

    results = []
    for f in npy_files:
        model_name = os.path.basename(f).replace(".npy", "")
        eigvals = np.load(f, allow_pickle=True).flatten()
        eigvals_abs = np.abs(eigvals)

        if np.any(~np.isfinite(eigvals_abs)):
            print(f"    ⚠️ 跳过 {model_name}（包含 NaN/Inf）")
            continue
        if len(eigvals_abs) == 0:
            print(f"    ⚠️ 跳过 {model_name}（空数组）")
            continue

        mode         = compute_kde_mode(eigvals_abs)
        zero_ratio   = compute_ratio(eigvals_abs, 0.0, 0.1)
        unit_ratio   = compute_ratio(eigvals_abs, 0.9, 1.1)
        growth_ratio = compute_ratio(eigvals_abs, 1.2, np.inf)   # ← 新增，闭区间

        results.append({
            "model":        model_name,
            "mode":         mode,
            "zero_ratio":   zero_ratio,
            "unit_ratio":   unit_ratio,
            "growth_ratio": growth_ratio,
            "n_eigvals":    len(eigvals_abs),
        })

        print(
            f"    {model_name:50s} | mode={mode:.6f} | "
            f"zero={zero_ratio:.1f}% | unit={unit_ratio:.1f}% | "
            f"growth={growth_ratio:.1f}% | n={len(eigvals_abs)}"
        )

    if not results:
        return None

    modes         = np.array([r["mode"]         for r in results])
    zero_ratios   = np.array([r["zero_ratio"]   for r in results])
    unit_ratios   = np.array([r["unit_ratio"]   for r in results])
    growth_ratios = np.array([r["growth_ratio"] for r in results])
    n_models      = len(results)

    # ── 三组两两配对 Wilcoxon ──────────────────────────────────────────
    w_zu, p_zu, d_zu, diff_zu = paired_wilcoxon(zero_ratios,   unit_ratios)
    w_zg, p_zg, d_zg, diff_zg = paired_wilcoxon(zero_ratios,   growth_ratios)
    w_ug, p_ug, d_ug, diff_ug = paired_wilcoxon(unit_ratios,   growth_ratios)

    # 主方向（zero vs unit，保持原有 dynamics 逻辑）
    if p_zu < 0.05 and diff_zu > 0:
        dynamics = "Collapse (zero > unit)"
    elif p_zu < 0.05 and diff_zu < 0:
        dynamics = "Fidelity (unit > zero)"
    else:
        dynamics = "No Preference"

    return {
        "modality":       modality_name,
        "n_models":       n_models,
        "mean_mode":      np.mean(modes),
        "std_mode":       np.std(modes, ddof=1) if n_models > 1 else 0.0,
        "mean_zero":      np.mean(zero_ratios),
        "mean_unit":      np.mean(unit_ratios),
        "mean_growth":    np.mean(growth_ratios),
        # zero vs unit
        "d_zu": d_zu, "w_zu": w_zu, "p_zu": p_zu, "diff_zu": diff_zu,
        # zero vs growth
        "d_zg": d_zg, "w_zg": w_zg, "p_zg": p_zg, "diff_zg": diff_zg,
        # unit vs growth
        "d_ug": d_ug, "w_ug": w_ug, "p_ug": p_ug, "diff_ug": diff_ug,
        "dynamics":       dynamics,
        "per_model":      results,
        "modes":          modes,
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


def print_intra_modal_table(summaries):
    print("\n" + "=" * 170)
    print("Table 1: Intra-Modal Paired Wilcoxon — Zero / Unit / Growth Ratios")
    print("  Regions: Zero=[0.0,0.1]  Unit=[0.9,1.1]  Growth=[1.2,+∞)")
    print("=" * 170)

    # ── 主表：各模态均值 ──
    hdr = (
        f"{'Modality':<12} | {'N':>4} | "
        f"{'Zero%':>6} | {'Unit%':>6} | {'Growth%':>7} | "
        f"{'Z-U diff':>9} | {'d(Z-U)':>7} | {'p(Z-U)':>10} | "
        f"{'Z-G diff':>9} | {'d(Z-G)':>7} | {'p(Z-G)':>10} | "
        f"{'U-G diff':>9} | {'d(U-G)':>7} | {'p(U-G)':>10} | "
        f"{'Dynamics':<25}"
    )
    print(hdr)
    print("-" * 170)

    for s in summaries:
        def dfmt(d): return f"{d:.2f}" if not np.isinf(d) else "inf"
        row = (
            f"{s['modality']:<12} | {s['n_models']:>4} | "
            f"{s['mean_zero']:>5.1f}% | {s['mean_unit']:>5.1f}% | {s['mean_growth']:>6.1f}% | "
            f"{s['diff_zu']:>+8.1f}% | {dfmt(s['d_zu']):>7} | {format_p(s['p_zu']):>10} | "
            f"{s['diff_zg']:>+8.1f}% | {dfmt(s['d_zg']):>7} | {format_p(s['p_zg']):>10} | "
            f"{s['diff_ug']:>+8.1f}% | {dfmt(s['d_ug']):>7} | {format_p(s['p_ug']):>10} | "
            f"{s['dynamics']:<25}"
        )
        print(row)

    print("=" * 170)


def print_inter_modal_table(summaries):
    if len(summaries) < 2:
        print("\n  ⚠️ 模态数不足 2，跳过模态间检验")
        return

    print("\n" + "=" * 100)
    print("Table 2: Inter-Modal Test — KDE Mode Comparison")
    print("=" * 100)

    if len(summaries) >= 3:
        h_stat, kw_p = kruskal(*[s["modes"] for s in summaries])
        print(f"  Kruskal-Wallis H = {h_stat:.4f}, p = {format_p(kw_p)}")

    print(f"\n  {'Comparison':<25} | {'U-stat':>10} | {'p-value':>12} | {'Direction':<35}")
    print(f"  {'-' * 90}")

    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            s1, s2 = summaries[i], summaries[j]
            u_stat, mw_p = mannwhitneyu(s1["modes"], s2["modes"], alternative='two-sided')
            direction = (
                f"{s1['modality']} < {s2['modality']}"
                if np.mean(s1["modes"]) < np.mean(s2["modes"])
                else f"{s1['modality']} > {s2['modality']}"
            )
            label = f"{s1['modality']} vs {s2['modality']}"
            print(f"  {label:<25} | {u_stat:>10.1f} | {format_p(mw_p):>12} | {direction} ({sig_star(mw_p)})")

    print("=" * 100)


def main():
    base_dir = "neweigvals"
    modalities = {
        "Vision":   os.path.join(base_dir, "vision"),
        "Audio":    os.path.join(base_dir, "audio"),
        "Language": os.path.join(base_dir, "language"),
    }

    summaries = []
    for name, path in modalities.items():
        print(f"\n{'=' * 60}\n  {name} ({path})\n{'=' * 60}")
        if not os.path.exists(path):
            print(f"  ⚠️ 路径不存在: {path}")
            continue
        s = analyze_modality(path, name)
        if s:
            summaries.append(s)

    if summaries:
        print_intra_modal_table(summaries)
        print_inter_modal_table(summaries)

    # 保存
    save_data = {}
    for s in summaries:
        k = s["modality"].lower()
        save_data[f"{k}_modes"]         = np.array([r["mode"]         for r in s["per_model"]])
        save_data[f"{k}_zero_ratios"]   = np.array([r["zero_ratio"]   for r in s["per_model"]])
        save_data[f"{k}_unit_ratios"]   = np.array([r["unit_ratio"]   for r in s["per_model"]])
        save_data[f"{k}_growth_ratios"] = np.array([r["growth_ratio"] for r in s["per_model"]])
        save_data[f"{k}_models"]        = np.array([r["model"]        for r in s["per_model"]])
    np.savez("dmd_spectral_summary.npz", **save_data)
    print("\n结果已保存到: dmd_spectral_summary.npz")


if __name__ == "__main__":
    main()