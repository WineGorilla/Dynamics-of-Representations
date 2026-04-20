"""
DMD 谱集中度分析
- 读取每个模态下所有模型的特征值
- 对每个模型：KDE 找峰值(mode)，算极点占比(zero-ratio)和单位圆占比(unit-ratio)
- 模态内检验：zero_ratio vs unit_ratio 配对检验（Wilcoxon）
- 模态间检验：Kruskal-Wallis + Mann-Whitney U 两两比较
- 输出表格
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import os
from glob import glob
from scipy.stats import gaussian_kde, wilcoxon, kruskal, mannwhitneyu


def compute_kde_mode(eigvals_abs, n_points=1000):
    """对特征值的绝对值做 KDE，找到峰值（众数）"""
    if len(eigvals_abs) < 3:
        return np.median(eigvals_abs)
    try:
        kde = gaussian_kde(eigvals_abs, bw_method='scott')
        x_grid = np.linspace(eigvals_abs.min(), eigvals_abs.max(), n_points)
        density = kde(x_grid)
        mode = x_grid[np.argmax(density)]
        return mode
    except Exception:
        return np.median(eigvals_abs)


def compute_ratio(eigvals_abs, low, high):
    """计算绝对值在 [low, high) 范围内的特征值占比"""
    return np.mean((eigvals_abs >= low) & (eigvals_abs < high)) * 100


def analyze_modality(modality_dir, modality_name):
    """分析一个模态下所有模型"""
    npy_files = sorted(glob(os.path.join(modality_dir, "*.npy")))
    if not npy_files:
        print(f"  ⚠️ {modality_name}: 没有找到 .npy 文件")
        return None

    results = []
    for f in npy_files:
        model_name = os.path.basename(f).replace(".npy", "")
        eigvals = np.load(f, allow_pickle=True)
        eigvals = eigvals.flatten()
        eigvals_abs = np.abs(eigvals)

        # 跳过包含 NaN 或 Inf 的模型
        if np.any(~np.isfinite(eigvals_abs)):
            print(f"    ⚠️ 跳过 {model_name}（包含 NaN/Inf）")
            continue

        if len(eigvals_abs) == 0:
            print(f"    ⚠️ 跳过 {model_name}（空数组）")
            continue

        mode = compute_kde_mode(eigvals_abs)
        zero_ratio = compute_ratio(eigvals_abs, 0.0, 0.1)
        unit_ratio = compute_ratio(eigvals_abs, 0.9, 1.1)

        results.append({
            "model": model_name,
            "mode": mode,
            "zero_ratio": zero_ratio,
            "unit_ratio": unit_ratio,
            "n_eigvals": len(eigvals_abs),
        })

        print(f"    {model_name:50s} | mode={mode:.6f} | zero={zero_ratio:.1f}% | unit={unit_ratio:.1f}% | n={len(eigvals_abs)}")

    if not results:
        return None

    # 模态级别汇总
    modes = np.array([r["mode"] for r in results])
    zero_ratios = np.array([r["zero_ratio"] for r in results])
    unit_ratios = np.array([r["unit_ratio"] for r in results])
    n_models = len(results)

    mean_mode = np.mean(modes)
    std_mode = np.std(modes, ddof=1) if n_models > 1 else 0.0
    mean_zero = np.mean(zero_ratios)
    mean_unit = np.mean(unit_ratios)

    # ── 模态内检验：zero_ratio vs unit_ratio 配对 Wilcoxon ──
    # H0: zero_ratio 和 unit_ratio 没有差异
    # H1: 有差异（双侧）
    diff = zero_ratios - unit_ratios
    mean_diff = np.mean(diff)

    if n_models >= 6 and not np.all(diff == 0):
        # Wilcoxon 要求至少 6 个样本且差值不能全为 0
        w_stat, w_pvalue = wilcoxon(zero_ratios, unit_ratios, alternative='two-sided')
    elif n_models >= 2 and not np.all(diff == 0):
        # 样本太少，退化为符号检验近似
        w_stat, w_pvalue = wilcoxon(zero_ratios, unit_ratios, alternative='two-sided')
    else:
        w_stat, w_pvalue = 0.0, 1.0

    # 效应量：配对 Cohen's d = mean(diff) / std(diff)
    std_diff = np.std(diff, ddof=1) if n_models > 1 else 1.0
    cohens_d_paired = mean_diff / std_diff if std_diff > 0 else float('inf')

    # 判断方向和动力学形态
    if w_pvalue < 0.05 and mean_diff > 0:
        dynamics = "Collapse (zero > unit)"
    elif w_pvalue < 0.05 and mean_diff < 0:
        dynamics = "Fidelity (unit > zero)"
    else:
        dynamics = "No Preference"

    summary = {
        "modality": modality_name,
        "n_models": n_models,
        "mean_mode": mean_mode,
        "std_mode": std_mode,
        "mean_zero_ratio": mean_zero,
        "mean_unit_ratio": mean_unit,
        "mean_diff": mean_diff,
        "cohens_d": cohens_d_paired,
        "w_stat": w_stat,
        "w_pvalue": w_pvalue,
        "dynamics": dynamics,
        "per_model": results,
        "modes": modes,
    }

    return summary


def format_p_value(p):
    if p < 0.0001:
        return "< 0.0001"
    elif p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return "< 0.01"
    elif p < 0.05:
        return "< 0.05"
    else:
        return f"{p:.4f}"


def print_intra_modal_table(summaries):
    """打印模态内检验表格"""
    print("\n" + "=" * 150)
    print("Table 1: Intra-Modal Test — Zero-Ratio vs Unit-Ratio (Paired Wilcoxon Signed-Rank Test)")
    print("  H0: zero_ratio = unit_ratio | H1: zero_ratio ≠ unit_ratio")
    print("=" * 150)

    header = (
        f"{'Modality':<12} | {'#Models':>7} | {'Mean Mode ± Std':>20} | "
        f"{'Zero [0,0.1)':>12} | {'Unit [0.9,1.1)':>14} | {'Diff (Z-U)':>10} | "
        f"{'Cohen d':>8} | {'W-stat':>8} | {'p-value':>12} | {'Dynamics'::<25}"
    )
    print(header)
    print("-" * 150)

    for s in summaries:
        mode_str = f"{s['mean_mode']:.4f} ± {s['std_mode']:.4f}"
        d_str = f"{s['cohens_d']:.2f}" if not np.isinf(s['cohens_d']) else "inf"
        p_str = format_p_value(s['w_pvalue'])

        row = (
            f"{s['modality']:<12} | {s['n_models']:>7} | {mode_str:>20} | "
            f"{s['mean_zero_ratio']:>11.1f}% | {s['mean_unit_ratio']:>13.1f}% | {s['mean_diff']:>+9.1f}% | "
            f"{d_str:>8} | {s['w_stat']:>8.1f} | {p_str:>12} | {s['dynamics']:<25}"
        )
        print(row)

    print("=" * 150)


def print_inter_modal_table(summaries):
    """打印模态间检验表格"""
    if len(summaries) < 2:
        print("\n  ⚠️ 模态数不足 2，跳过模态间检验")
        return

    print("\n" + "=" * 100)
    print("Table 2: Inter-Modal Test — KDE Mode Comparison")
    print("=" * 100)

    # Kruskal-Wallis (三组及以上)
    if len(summaries) >= 3:
        all_modes = [s["modes"] for s in summaries]
        h_stat, kw_pvalue = kruskal(*all_modes)
        print(f"  Kruskal-Wallis H = {h_stat:.4f}, p = {format_p_value(kw_pvalue)}")
    elif len(summaries) == 2:
        kw_pvalue = None
        print("  只有 2 个模态，跳过 Kruskal-Wallis，直接 Mann-Whitney U")

    # 两两 Mann-Whitney U
    print(f"\n  {'Comparison':<25} | {'U-stat':>10} | {'p-value':>12} | {'Direction':<30}")
    print(f"  {'-'*85}")

    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            s1 = summaries[i]
            s2 = summaries[j]
            u_stat, mw_pvalue = mannwhitneyu(
                s1["modes"], s2["modes"], alternative='two-sided'
            )
            # 方向
            if np.mean(s1["modes"]) < np.mean(s2["modes"]):
                direction = f"{s1['modality']} < {s2['modality']}"
            else:
                direction = f"{s1['modality']} > {s2['modality']}"

            label = f"{s1['modality']} vs {s2['modality']}"
            sig = "***" if mw_pvalue < 0.001 else "**" if mw_pvalue < 0.01 else "*" if mw_pvalue < 0.05 else "n.s."

            print(f"  {label:<25} | {u_stat:>10.1f} | {format_p_value(mw_pvalue):>12} | {direction} ({sig})")

    print("=" * 100)


def main():
    # ========== 修改这里为你的路径 ==========
    base_dir = "processed_new/eigvals"
    modalities = {
        "Vision": os.path.join(base_dir, "vision"),
        "Audio": os.path.join(base_dir, "audio"),
        "Language": os.path.join(base_dir, "language_new"),
    }

    summaries = []

    for modality_name, modality_dir in modalities.items():
        print(f"\n{'=' * 60}")
        print(f"  {modality_name} ({modality_dir})")
        print(f"{'=' * 60}")

        if not os.path.exists(modality_dir):
            print(f"  ⚠️ 路径不存在: {modality_dir}")
            continue

        summary = analyze_modality(modality_dir, modality_name)
        if summary:
            summaries.append(summary)

    if summaries:
        # 模态内检验
        print_intra_modal_table(summaries)
        # 模态间检验
        print_inter_modal_table(summaries)

    # 保存结果
    save_path = "dmd_spectral_summary.npz"
    save_data = {}
    for s in summaries:
        key = s["modality"].lower()
        save_data[f"{key}_modes"] = np.array([r["mode"] for r in s["per_model"]])
        save_data[f"{key}_zero_ratios"] = np.array([r["zero_ratio"] for r in s["per_model"]])
        save_data[f"{key}_unit_ratios"] = np.array([r["unit_ratio"] for r in s["per_model"]])
        save_data[f"{key}_models"] = np.array([r["model"] for r in s["per_model"]])
    np.savez(save_path, **save_data)
    print(f"\n结果已保存到: {save_path}")


if __name__ == "__main__":
    main()