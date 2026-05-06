"""
架构间 DMD 特征值对比分析
=========================
三个模态按架构分组，对比 KDE Mode / Mean|λ| / Zero% / Unit% / Growth%

分组:
  Vision (5组):  ViT / CLIP / Swin / SAM / CNN
  Audio  (5组):  Wav2Vec2 / HuBERT / WavLM / Data2Vec-Audio / Whisper / SEW / UniSpeech
  Language (5组): BERT / RoBERTa / ALBERT / ELECTRA / DistilBERT / 其他

用法:
  python compare_architectures.py
"""

import numpy as np
import os
from glob import glob
from scipy.stats import gaussian_kde, kruskal, mannwhitneyu


# ####################################################################
#  架构分组定义
# ####################################################################

VISION_GROUPS = {
    "ViT": [
        "beit-base-patch16-224-pt22k-ft22k", "beit-large-patch16-224-pt22k-ft22k",
        "data2vec-vision-base", "data2vec-vision-large",
        "deit-base-patch16-224", "deit-small-patch16-224",
        "dino-vitb16", "dino-vits16",
        "dinov2-base", "dinov2-large", "dinov2-small",
        "vit-base-patch16-224-in21k", "vit-large-patch16-224-in21k",
        "vit-mae-base", "vit-mae-large",
        "vit-msn-base", "vit-msn-large",
    ],
    "CLIP": [
        "clip-vit-base-patch32", "clip-vit-base-patch16", "clip-vit-large-patch14",
    ],
    "Swin": [
        "swin-tiny-patch4-window7-224", "swin-small-patch4-window7-224",
        "swin-large-patch4-window7-224",
    ],
    "SAM": [
        "sam-vit-base", "sam-vit-large", "sam-vit-huge",
    ],
    "CNN": [
        "resnet50", "resnet101",
        "densenet121", "densenet201",
        "efficientnet_b0", "efficientnet_b4",
        "convnext_tiny", "convnext_base",
        "vgg16", "vgg19",
    ],
}

AUDIO_GROUPS = {
    "Wav2Vec2": [
        "wav2vec2-base", "wav2vec2-base-960h", "wav2vec2-large", "wav2vec2-large-960h",
        "wav2vec2-large-xlsr-53", "wav2vec2-xls-r-300m", "wav2vec2-xls-r-1b",
        "wav2vec2-base-superb-ks",
        "wav2vec2-conformer-rel-pos-large", "wav2vec2-conformer-rope-large-960h-ft",
        "w2v-bert-2.0",
    ],
    "HuBERT": [
        "hubert-base-ls960", "hubert-base-superb-ks",
        "hubert-large-ls960-ft", "hubert-xlarge-ls960-ft",
    ],
    "WavLM": [
        "wavlm-base", "wavlm-base-plus", "wavlm-large",
    ],
    "Data2Vec": [
        "data2vec-audio-base", "data2vec-audio-base-960h",
        "data2vec-audio-large", "data2vec-audio-large-960h",
    ],
    "SEW": [
        "sew-tiny-100k", "sew-small-100k", "sew-mid-100k",
        "sew-d-tiny-100k", "sew-d-small-100k", "sew-d-mid-100k",
    ],
    "UniSpeech": [
        "unispeech-large-1500h-cv",
        "unispeech-sat-base", "unispeech-sat-base-plus", "unispeech-sat-large",
    ],
    "Whisper": [
        "whisper-tiny", "whisper-base", "whisper-small", "whisper-medium",
    ],
}

LANGUAGE_GROUPS = {
    "BERT": [
        "bert-base-cased", "bert-base-uncased", "bert-large-cased", "bert-large-uncased",
        "bert-base-multilingual-cased",
    ],
    "RoBERTa": [
        "roberta-base", "roberta-large",
        "distilroberta-base",
        "xlm-roberta-base", "xlm-roberta-large",
        "camembert-base",
        "ibert-roberta-base",
    ],
    "ALBERT": [
        "albert-base-v2", "albert-large-v2", "albert-xlarge-v2",
    ],
    "ELECTRA": [
        "electra-base-discriminator", "electra-large-discriminator",
        "electra-small-discriminator",
    ],
    "DistilBERT": [
        "distilbert-base-uncased", "distilbert-base-multilingual-cased",
    ],
    "Other": [
        "MiniLM-L6-H384-uncased", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "mpnet-base",
        "conv-bert-base", "conv-bert-medium-small",
        "data2vec-text-base",
        "deberta-base", "deberta-large",
        "ernie-2.0-base-en", "ernie-2.0-large-en",
        "rembert",
        "squeezebert-uncased",
        "t5-small",
        "xlnet-base-cased", "xlnet-large-cased",
    ],
}

MODALITY_CONFIG = {
    "Vision":   {"dir": "neweigvals/vision",   "groups": VISION_GROUPS},
    "Audio":    {"dir": "neweigvals/audio",     "groups": AUDIO_GROUPS},
    "Language": {"dir": "neweigvals/language",  "groups": LANGUAGE_GROUPS},
}


# ####################################################################
#  统计工具
# ####################################################################

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


# ####################################################################
#  分组分析
# ####################################################################

def analyze_group(eigvals_dir, model_names, group_name):
    """分析一个架构组的所有模型"""
    results = []

    for model_name in model_names:
        npy_path = os.path.join(eigvals_dir, f"{model_name}.npy")
        if not os.path.exists(npy_path):
            continue

        eigvals = np.load(npy_path, allow_pickle=True).flatten()
        eigvals_abs = np.abs(eigvals)

        if np.any(~np.isfinite(eigvals_abs)) or len(eigvals_abs) == 0:
            continue

        results.append({
            "model":           model_name,
            "mode":            compute_kde_mode(eigvals_abs),
            "mean_abs_lambda": float(np.mean(eigvals_abs)),
            "zero_ratio":      compute_ratio(eigvals_abs, 0.0, 0.1),
            "unit_ratio":      compute_ratio(eigvals_abs, 0.9, 1.1),
            "growth_ratio":    compute_ratio(eigvals_abs, 1.2, np.inf),
            "n_eigvals":       len(eigvals_abs),
        })

    if not results:
        return None

    return {
        "group":   group_name,
        "n":       len(results),
        "modes":   np.array([r["mode"] for r in results]),
        "means":   np.array([r["mean_abs_lambda"] for r in results]),
        "zeros":   np.array([r["zero_ratio"] for r in results]),
        "units":   np.array([r["unit_ratio"] for r in results]),
        "growths": np.array([r["growth_ratio"] for r in results]),
        "per_model": results,
    }


def run_modality(modality_name, eigvals_dir, groups):
    """对一个模态的所有架构组进行分析和检验"""

    print(f"\n{'#'*100}")
    print(f"  {modality_name}")
    print(f"{'#'*100}")

    group_results = []

    for group_name, model_names in groups.items():
        g = analyze_group(eigvals_dir, model_names, group_name)
        if g is None:
            print(f"\n  ⚠️ {group_name}: 无数据")
            continue
        group_results.append(g)

        # Per-model 详情
        print(f"\n  {group_name} ({g['n']} models):")
        for r in g["per_model"]:
            print(f"    {r['model']:45s} | mode={r['mode']:.4f} | mean|λ|={r['mean_abs_lambda']:.4f} | "
                  f"zero={r['zero_ratio']:.1f}% | unit={r['unit_ratio']:.1f}% | growth={r['growth_ratio']:.1f}%")

    if len(group_results) < 2:
        print(f"\n  ⚠️ 分组不足 2，跳过检验")
        return group_results

    # ── Table 1: 各组汇总 ──
    print(f"\n  {'='*110}")
    print(f"  Table: Architecture Group Summary — {modality_name}")
    print(f"  {'='*110}")
    print(f"  {'Group':<15} | {'N':>3} | {'Mode':>10} | {'Mean|λ|':>10} | "
          f"{'Zero%':>8} | {'Unit%':>8} | {'Growth%':>8}")
    print(f"  {'-'*110}")

    for g in group_results:
        print(f"  {g['group']:<15} | {g['n']:>3} | "
              f"{np.mean(g['modes']):>10.4f} | {np.mean(g['means']):>10.4f} | "
              f"{np.mean(g['zeros']):>7.1f}% | {np.mean(g['units']):>7.1f}% | "
              f"{np.mean(g['growths']):>7.1f}%")

    print(f"  {'='*110}")

    # ── Kruskal-Wallis 全局检验 ──
    if len(group_results) >= 3:
        print(f"\n  Kruskal-Wallis 全局检验:")
        for metric_name, key in [("Mode", "modes"), ("Mean|λ|", "means"),
                                  ("Zero%", "zeros"), ("Unit%", "units"), ("Growth%", "growths")]:
            arrays = [g[key] for g in group_results if len(g[key]) >= 2]
            if len(arrays) >= 3:
                h_stat, kw_p = kruskal(*arrays)
                print(f"    {metric_name:<10}: H={h_stat:.4f}, p={format_p(kw_p)} {sig_star(kw_p)}")

    # ── 两两 Mann-Whitney U ──
    print(f"\n  两两对比 (Mann-Whitney U):")
    print(f"  {'Comparison':<30} | {'Mode p':>12} | {'Mean|λ| p':>12} | "
          f"{'Zero% p':>12} | {'Unit% p':>12} | {'Growth% p':>12}")
    print(f"  {'-'*110}")

    for i in range(len(group_results)):
        for j in range(i + 1, len(group_results)):
            g1, g2 = group_results[i], group_results[j]
            label = f"{g1['group']} vs {g2['group']}"

            p_vals = []
            for key in ["modes", "means", "zeros", "units", "growths"]:
                if len(g1[key]) >= 2 and len(g2[key]) >= 2:
                    _, p = mannwhitneyu(g1[key], g2[key], alternative='two-sided')
                    p_vals.append(p)
                else:
                    p_vals.append(float('nan'))

            p_strs = [f"{format_p(p):>12}" for p in p_vals]
            sigs = [sig_star(p) for p in p_vals]

            print(f"  {label:<30} | {p_strs[0]} {sigs[0]:>4} | {p_strs[1]} {sigs[1]:>4} | "
                  f"{p_strs[2]} {sigs[2]:>4} | {p_strs[3]} {sigs[3]:>4} | {p_strs[4]} {sigs[4]:>4}")

    print(f"  {'='*110}")

    return group_results


# ####################################################################
#  Main
# ####################################################################

def main():
    all_results = {}

    for modality_name, cfg in MODALITY_CONFIG.items():
        eigvals_dir = cfg["dir"]
        groups = cfg["groups"]

        if not os.path.exists(eigvals_dir):
            print(f"\n  ⚠️ {modality_name}: 路径不存在 {eigvals_dir}")
            continue

        results = run_modality(modality_name, eigvals_dir, groups)
        all_results[modality_name] = results

    # ── 保存 ──
    save_data = {}
    for mod_name, groups in all_results.items():
        if groups is None:
            continue
        for g in groups:
            prefix = f"{mod_name.lower()}_{g['group'].lower()}"
            save_data[f"{prefix}_modes"] = g["modes"]
            save_data[f"{prefix}_means"] = g["means"]
            save_data[f"{prefix}_zeros"] = g["zeros"]
            save_data[f"{prefix}_units"] = g["units"]
            save_data[f"{prefix}_growths"] = g["growths"]
            save_data[f"{prefix}_models"] = np.array([r["model"] for r in g["per_model"]])

    np.savez("architecture_comparison.npz", **save_data)
    print("\n结果已保存到: architecture_comparison.npz")
    print("\n✓ 全部完成！")


if __name__ == "__main__":
    main()