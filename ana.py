"""
DMD 谱集中度分析
- 读取每个模态下所有模型的特征值
- 对每个模型：KDE 找峰值(mode)，算极点占比(zero-ratio)和单位圆占比(unit-ratio)
- 对每个模态：汇总统计，Cohen's d，T 检验
- 输出表格
"""

import numpy as np
import os
from glob import glob
from scipy.stats import gaussian_kde, ttest_1samp


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


def compute_zero_ratio(eigvals_abs, low=0.0, high=0.1):
    """计算绝对值在 [low, high) 范围内的特征值占比"""
    return np.mean((eigvals_abs >= low) & (eigvals_abs < high)) * 100


def compute_unit_ratio(eigvals_abs, low=0.9, high=1.1):
    """计算绝对值在 [low, high) 范围内的特征值占比（靠近单位圆）"""
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
        zero_ratio = compute_zero_ratio(eigvals_abs, low=0.0, high=0.1)
        unit_ratio = compute_unit_ratio(eigvals_abs, low=0.9, high=1.1)

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

    # Cohen's d: (threshold - mean) / std
    threshold = 0.05
    cohens_d = (threshold - mean_mode) / std_mode if std_mode > 0 else float('inf')

    # 单样本 T 检验: H0: mean_mode = threshold
    if n_models > 1:
        t_stat, p_value = ttest_1samp(modes, threshold)
    else:
        t_stat, p_value = 0.0, 1.0

    # 判断动力学形态
    if mean_mode < 0.01 and mean_zero > 80:
        dynamics = "Strong Collapse"
    elif mean_mode < 0.1 and mean_zero > 40:
        dynamics = "Moderate Decay"
    elif mean_mode > 0.5 and mean_unit > 30:
        dynamics = "High Fidelity"
    elif mean_unit > 50:
        dynamics = "Near-Critical"
    else:
        dynamics = "Mixed"

    summary = {
        "modality": modality_name,
        "n_models": n_models,
        "mean_mode": mean_mode,
        "std_mode": std_mode,
        "mean_zero_ratio": mean_zero,
        "mean_unit_ratio": mean_unit,
        "cohens_d": cohens_d,
        "t_stat": t_stat,
        "p_value": p_value,
        "dynamics": dynamics,
        "per_model": results,
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


def print_summary_table(summaries):
    """打印汇总表格"""
    print("\n" + "=" * 140)
    print("Table 1: Cross-Modal Comparison of Feature Collapse via DMD Spectral Concentration")
    print("=" * 140)

    header = (
        f"{'Modality':<12} | {'#Models':>7} | {'Mean Mode ± Std':>20} | "
        f"{'Zero [0,0.1)':>12} | {'Unit [0.9,1.1)':>14} | "
        f"{'Cohen d':>8} | {'p-value':>12} | {'Dynamics Type':<20}"
    )
    print(header)
    print("-" * 140)

    for s in summaries:
        mode_str = f"{s['mean_mode']:.4f} ± {s['std_mode']:.4f}"
        d_str = f"{s['cohens_d']:.2f}" if not np.isinf(s['cohens_d']) else "inf"
        p_str = format_p_value(s['p_value'])

        row = (
            f"{s['modality']:<12} | {s['n_models']:>7} | {mode_str:>20} | "
            f"{s['mean_zero_ratio']:>11.1f}% | {s['mean_unit_ratio']:>13.1f}% | "
            f"{d_str:>8} | {p_str:>12} | {s['dynamics']:<20}"
        )
        print(row)

    print("=" * 140)


def main():
    # ========== 修改这里为你的路径 ==========
    base_dir = "processed_new/eigvals"
    modalities = {
        "Vision": os.path.join(base_dir, "vision"),
        "Audio": os.path.join(base_dir, "audio"),
        "Language": os.path.join(base_dir, "language"),
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
        print_summary_table(summaries)

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