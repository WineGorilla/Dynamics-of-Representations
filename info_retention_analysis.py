"""
Information Retention Across Dynamical Modes
=============================================
统一统计分析框架：基于 context ablation 实验

核心假设：
  稳态模式（center≈1.0）编码全局结构，对 context 破坏更鲁棒
  瞬态模式（center≈0.0）编码局部细节，对 context 破坏更敏感

实验设计：
  对每个模型的每个刺激：
    1. 完整输入 → per-layer embeddings → soft DMD fusion (各 center)
    2. 破坏输入 → per-layer embeddings → soft DMD fusion (各 center)
    3. cos(fused_full, fused_masked) = 信息保留度
  
  如果 sim(center=1.0) > sim(center=0.0)，说明稳态模式对破坏更鲁棒

统计检验：
  Level 1 — 模型内：配对 Wilcoxon (per-stimulus paired comparison)
  Level 2 — 模态内：所有模型的 effect 是否一致 > 0
  Level 3 — 跨模态：Kruskal-Wallis + Mann-Whitney U

用法：
  # 先跑三个模态的 ablation（复用你已有的脚本），保存 per-sample similarity
  # 然后运行本脚本做统计分析
  
  python info_retention_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from scipy.stats import wilcoxon, kruskal, mannwhitneyu

# ═══════════════════════════════════════════════════════════════
#  导入各模态的 ablation 模块
# ═══════════════════════════════════════════════════════════════

from core.dmd import fuse_layers_single_soft_dmd


# ═══════════════════════════════════════════════════════════════
#  通用工具
# ═══════════════════════════════════════════════════════════════

def cosine_sim_batch(A, B):
    """(N, d) × 2 → (N,) per-sample cosine similarity"""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (A_norm * B_norm).sum(axis=1)


def dmd_fuse_all(X_LNd, k=3, center=1.0, sigma=0.1):
    """(L, N, d) → (N, d) via soft DMD fusion per sample"""
    L, N, d = X_LNd.shape
    fused = np.zeros((N, d), dtype=np.float32)
    for n in range(N):
        fused[n] = fuse_layers_single_soft_dmd(X_LNd[:, n, :], r=k, center=center, sigma=sigma)
    return fused


def format_p(p):
    if p < 0.0001:   return "< 0.0001"
    elif p < 0.001:  return "< 0.001"
    elif p < 0.01:   return "< 0.01"
    elif p < 0.05:   return "< 0.05"
    else:            return f"{p:.4f}"


# ═══════════════════════════════════════════════════════════════
#  Vision 模块
# ═══════════════════════════════════════════════════════════════

def collect_vision_data(model_name, img_root, device="cuda", batch_size=8, 
                        max_samples=500, keep_ratio=0.5):
    """
    返回 X_full, X_masked: (L, N, d)
    """
    from PIL import Image
    from core.encoder.image_encoder import load_image_model, get_image_embeddings_from_pil
    
    # 随机 patch 保留
    def random_patch_keep_pil(img, ratio, patch_size=16):
        img = img.copy()
        W, H = img.size
        patches_x = W // patch_size
        patches_y = H // patch_size
        n_patches = patches_x * patches_y
        n_keep = max(1, int(n_patches * ratio))
        keep_idx = set(np.random.choice(n_patches, size=n_keep, replace=False))
        pixels = img.load()
        for idx in range(n_patches):
            if idx not in keep_idx:
                px = (idx % patches_x) * patch_size
                py = (idx // patches_x) * patch_size
                for x in range(px, min(px + patch_size, W)):
                    for y in range(py, min(py + patch_size, H)):
                        pixels[x, y] = (0, 0, 0)
        return img
    
    # 收集图像
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG"):
        img_paths.extend(glob(os.path.join(img_root, "**", ext), recursive=True))
    img_paths = sorted(set(img_paths))[:max_samples]
    print(f"    {len(img_paths)} images")
    
    full_imgs = [Image.open(p).convert("RGB") for p in img_paths]
    masked_imgs = [random_patch_keep_pil(img, keep_ratio) for img in full_imgs]
    
    extractor, model = load_image_model(model_name, device=device)
    
    X_full_layers = get_image_embeddings_from_pil(
        extractor, model, full_imgs, device=device, cls_only=True, batch_size=batch_size)
    X_mask_layers = get_image_embeddings_from_pil(
        extractor, model, masked_imgs, device=device, cls_only=True, batch_size=batch_size)
    
    X_full = np.stack(X_full_layers, axis=0).astype(np.float32)
    X_mask = np.stack(X_mask_layers, axis=0).astype(np.float32)
    return X_full, X_mask


# ═══════════════════════════════════════════════════════════════
#  Audio 模块
# ═══════════════════════════════════════════════════════════════

def collect_audio_data(model_name, audio_dir, device="cuda", 
                       max_samples=50, keep_ratio=0.5, sr=16000):
    """
    返回 X_full, X_masked: (L, N, d)
    """
    import librosa
    import torch
    from core.encoder.audio_encoder import load_audio_model
    
    def random_frame_keep(y, ratio, frame_sec=0.02, sr=16000):
        frame_size = max(1, int(sr * frame_sec))
        n_frames = len(y) // frame_size
        n_keep = max(1, int(n_frames * ratio))
        keep_idx = set(np.random.choice(n_frames, size=n_keep, replace=False))
        y_out = np.zeros_like(y)
        for idx in keep_idx:
            start = idx * frame_size
            y_out[start:start + frame_size] = y[start:start + frame_size]
        return y_out
    
    def embed_audio(y, processor, model, device, sr=16000, tr=2.0):
        chunk_size = int(sr * tr)
        chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]
        n_layers = model.config.num_hidden_layers + 1
        layer_accum = [[] for _ in range(n_layers)]
        for chunk in chunks:
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            inputs = processor(chunk, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            for l, h in enumerate(outputs.hidden_states):
                emb = h.mean(dim=1).squeeze(0).cpu().numpy()
                layer_accum[l].append(emb)
        X = np.stack([np.stack(layer_accum[l]).mean(axis=0) for l in range(n_layers)])
        return X.astype(np.float32)
    
    audio_paths = sorted(glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))
    audio_paths += sorted(glob(os.path.join(audio_dir, "**", "*.mp3"), recursive=True))
    audio_paths = audio_paths[:max_samples]
    print(f"    {len(audio_paths)} audio files")
    
    processor, model = load_audio_model(model_name, device=device)
    
    full_embeds, mask_embeds = [], []
    for path in tqdm(audio_paths, desc="    Encoding audio"):
        y, _ = librosa.load(path, sr=sr, mono=True)
        y_mask = random_frame_keep(y, keep_ratio, sr=sr)
        full_embeds.append(embed_audio(y, processor, model, device, sr))
        mask_embeds.append(embed_audio(y_mask, processor, model, device, sr))
    
    X_full = np.stack(full_embeds, axis=1).astype(np.float32)
    X_mask = np.stack(mask_embeds, axis=1).astype(np.float32)
    return X_full, X_mask


# ═══════════════════════════════════════════════════════════════
#  Language 模块
# ═══════════════════════════════════════════════════════════════

def collect_language_data(model_name, csv_path, device="cuda", keep_ratio=0.5):
    """
    返回 X_full, X_masked: (L, N, d)
    """
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    def random_word_keep(words, ratio, mask_token="[MASK]"):
        keep = max(1, int(len(words) * ratio))
        keep_idx = set(np.random.choice(len(words), size=keep, replace=False))
        return [words[i] if i in keep_idx else mask_token for i in range(len(words))]
    
    def get_cls(sentence, tokenizer, model, device):
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return np.stack([h[:, 0, :].squeeze(0).cpu().numpy() for h in outputs.hidden_states]).astype(np.float32)
    
    df = pd.read_csv(csv_path).sort_values(["section", "onset"])
    sec = sorted(df["section"].unique())[0]
    words = df[df["section"] == sec]["word"].dropna().astype(str).tolist()
    words = [w for w in words if w.strip() and w != "nan"]
    print(f"    {len(words)} words")
    
    # 构建窗口
    win_size, step = 50, 25
    sentences = []
    for i in range(0, max(1, len(words) - win_size + 1), step):
        sentences.append(words[i:i + win_size])
    if not sentences:
        sentences = [words]
    print(f"    {len(sentences)} sentence windows")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()
    
    full_layers, mask_layers = [], []
    for sent_words in tqdm(sentences, desc="    Encoding language"):
        full_layers.append(get_cls(" ".join(sent_words), tokenizer, model, device))
        masked = random_word_keep(sent_words, keep_ratio)
        mask_layers.append(get_cls(" ".join(masked), tokenizer, model, device))
    
    X_full = np.stack(full_layers, axis=1).astype(np.float32)
    X_mask = np.stack(mask_layers, axis=1).astype(np.float32)
    return X_full, X_mask


# ═══════════════════════════════════════════════════════════════
#  核心分析：单个模型
# ═══════════════════════════════════════════════════════════════

def dmd_fuse_random_modes(X_LNd, k=3, n_repeats=20):
    """
    Random baseline: DMD 分解后随机选 r 个模式（均匀权重）做重构，
    重复 n_repeats 次取均值，消除随机性。
    
    (L, N, d) → (N, d)
    """
    from core.dmd import compute_dmd_eigenvalues  # 只用来获取分解
    from scipy.linalg import svd as scipy_svd
    
    L, N, d = X_LNd.shape
    fused_accum = np.zeros((N, d), dtype=np.float64)
    
    for n in range(N):
        X = X_LNd[:, n, :].astype(np.float64)  # (L, d)
        X1 = X[:-1].T  # (d, L-1)
        X2 = X[1:].T   # (d, L-1)
        
        # DMD 分解
        U, S, Vt = np.linalg.svd(X1, full_matrices=False)
        tol = 1e-10 * S[0] if len(S) > 0 else 1e-10
        r = int(np.sum(S > tol))
        if r == 0:
            continue
        r = min(r, k) if k > 0 else r
        
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]
        
        A_r = U_r.T @ X2 @ Vt_r.T @ np.diag(1.0 / S_r)
        eigvals, W = np.linalg.eig(A_r)
        Phi = X2 @ Vt_r.T @ np.diag(1.0 / S_r) @ W  # (d, r)
        
        # 模式振幅
        try:
            b = np.linalg.lstsq(Phi, X[0], rcond=None)[0]
        except np.linalg.LinAlgError:
            b = np.zeros(len(eigvals), dtype=complex)
        
        n_steps = L - 1
        n_modes = len(eigvals)
        
        # 随机选模式，重复 n_repeats 次取均值
        sample_fused = np.zeros(d, dtype=np.float64)
        n_select = max(1, n_modes // 2)  # 随机选一半的模式
        
        for _ in range(n_repeats):
            # 随机选模式子集，均匀权重
            selected = np.random.choice(n_modes, size=n_select, replace=False)
            weights = np.zeros(n_modes)
            weights[selected] = 1.0
            
            coeffs = weights * b * (eigvals ** n_steps)
            x_hat = Phi @ coeffs
            sample_fused += x_hat.real
        
        fused_accum[n] = sample_fused / n_repeats
    
    return fused_accum.astype(np.float32)


def analyze_one_model(X_full, X_masked, model_name, centers, k=3, sigma=0.1, n_random_repeats=20):
    """
    对一个模型：用不同 center DMD fusion，比较 full vs masked 的 cosine sim。
    包含 random baseline：随机选模式子集做重构。
    
    Returns
    -------
    result : dict
        包含 per-center 的 per-sample similarity 和统计检验结果
    """
    L, N, d = X_full.shape
    print(f"    {model_name}: L={L}, N={N}, d={d}")
    
    # ── 每个 center 的 per-sample similarity ──
    per_center_sims = {}
    for c in centers:
        fused_full = dmd_fuse_all(X_full, k=k, center=c, sigma=sigma)
        fused_mask = dmd_fuse_all(X_masked, k=k, center=c, sigma=sigma)
        sims = cosine_sim_batch(fused_full, fused_mask)
        per_center_sims[c] = sims
        print(f"      center={c:.1f}: sim={sims.mean():.4f} ± {sims.std():.4f}")
    
    # ── Random baseline ──
    print(f"      [random baseline] computing ({n_random_repeats} repeats)...")
    rand_fused_full = dmd_fuse_random_modes(X_full, k=k, n_repeats=n_random_repeats)
    rand_fused_mask = dmd_fuse_random_modes(X_masked, k=k, n_repeats=n_random_repeats)
    sims_random = cosine_sim_batch(rand_fused_full, rand_fused_mask)
    print(f"      random:    sim={sims_random.mean():.4f} ± {sims_random.std():.4f}")
    
    # ── 核心检验 1：center=1.0 vs center=0.0 配对 Wilcoxon ──
    c_low = centers[0]    # 瞬态 (0.0)
    c_high = centers[-1]  # 稳态 (1.0)
    # 找最接近 0.5 的 center 作为中间条件
    c_mid = min(centers, key=lambda x: abs(x - 0.5))
    
    sims_low = per_center_sims[c_low]
    sims_mid = per_center_sims[c_mid]
    sims_high = per_center_sims[c_high]
    
    def _paired_wilcoxon(a, b, N):
        """单侧 Wilcoxon: H1: b > a"""
        diff = b - a
        if N >= 2 and not np.all(diff == 0):
            w, p = wilcoxon(b, a, alternative='greater')
        else:
            w, p = 0.0, 1.0
        mean_d = np.mean(diff)
        std_d = np.std(diff, ddof=1) if N > 1 else 1.0
        d = mean_d / std_d if std_d > 0 else float('inf')
        return w, p, mean_d, d
    
    # S vs T
    w_st, p_st, diff_st, d_st = _paired_wilcoxon(sims_low, sims_high, N)
    sig_st = "***" if p_st < 0.001 else "**" if p_st < 0.01 else "*" if p_st < 0.05 else "n.s."
    print(f"      Δ(steady-transient) = {diff_st:+.4f}, d={d_st:.2f}, p={format_p(p_st)} {sig_st}")
    
    # S vs M
    w_sm, p_sm, diff_sm, d_sm = _paired_wilcoxon(sims_mid, sims_high, N)
    sig_sm = "***" if p_sm < 0.001 else "**" if p_sm < 0.01 else "*" if p_sm < 0.05 else "n.s."
    print(f"      Δ(steady-mid)       = {diff_sm:+.4f}, d={d_sm:.2f}, p={format_p(p_sm)} {sig_sm}")
    
    # M vs T
    w_mt, p_mt, diff_mt, d_mt = _paired_wilcoxon(sims_low, sims_mid, N)
    sig_mt = "***" if p_mt < 0.001 else "**" if p_mt < 0.01 else "*" if p_mt < 0.05 else "n.s."
    print(f"      Δ(mid-transient)    = {diff_mt:+.4f}, d={d_mt:.2f}, p={format_p(p_mt)} {sig_mt}")
    
    # S vs R
    w_sr, p_sr, diff_sr, d_sr = _paired_wilcoxon(sims_random, sims_high, N)
    sig_sr = "***" if p_sr < 0.001 else "**" if p_sr < 0.01 else "*" if p_sr < 0.05 else "n.s."
    print(f"      Δ(steady-random)    = {diff_sr:+.4f}, d={d_sr:.2f}, p={format_p(p_sr)} {sig_sr}")
    
    # 单调性判断: T < M < S ?
    monotonic = (np.mean(sims_low) < np.mean(sims_mid) < np.mean(sims_high))
    if monotonic:
        print(f"      ✓ Monotonic: sim(T) < sim(M) < sim(S)")
    else:
        print(f"      ✗ Non-monotonic: sim(T)={np.mean(sims_low):.4f}, sim(M)={np.mean(sims_mid):.4f}, sim(S)={np.mean(sims_high):.4f}")
    
    return {
        "model": model_name,
        "n_stimuli": N,
        "per_center_sims": per_center_sims,
        "center_means": {c: float(np.mean(s)) for c, s in per_center_sims.items()},
        "c_mid": c_mid,
        "monotonic": monotonic,
        # steady vs transient
        "mean_diff": diff_st,
        "cohens_d": d_st,
        "w_stat": w_st,
        "p_value": p_st,
        # steady vs mid
        "mean_diff_sm": diff_sm,
        "cohens_d_sm": d_sm,
        "w_stat_sm": w_sm,
        "p_value_sm": p_sm,
        # mid vs transient
        "mean_diff_mt": diff_mt,
        "cohens_d_mt": d_mt,
        "w_stat_mt": w_mt,
        "p_value_mt": p_mt,
        # steady vs random
        "random_sim_mean": float(sims_random.mean()),
        "mean_diff_sr": diff_sr,
        "cohens_d_sr": d_sr,
        "w_stat_sr": w_sr,
        "p_value_sr": p_sr,
    }


# ═══════════════════════════════════════════════════════════════
#  模态级分析 + 跨模态检验
# ═══════════════════════════════════════════════════════════════

def aggregate_modality(model_results, modality_name):
    """模态级汇总：所有模型的 mean_diff 是否一致 > 0"""
    all_diffs = np.array([r["mean_diff"] for r in model_results])
    all_d = np.array([r["cohens_d"] for r in model_results])
    all_diffs_sr = np.array([r["mean_diff_sr"] for r in model_results])
    all_d_sr = np.array([r["cohens_d_sr"] for r in model_results])
    all_diffs_sm = np.array([r["mean_diff_sm"] for r in model_results])
    all_diffs_mt = np.array([r["mean_diff_mt"] for r in model_results])
    n_monotonic = sum(1 for r in model_results if r["monotonic"])
    n = len(all_diffs)
    
    def _agg_wilcoxon(diffs):
        if len(diffs) >= 2 and not np.all(diffs == 0):
            return wilcoxon(diffs, alternative='greater')
        return 0.0, 1.0
    
    w_agg, p_agg = _agg_wilcoxon(all_diffs)
    w_agg_sr, p_agg_sr = _agg_wilcoxon(all_diffs_sr)
    w_agg_sm, p_agg_sm = _agg_wilcoxon(all_diffs_sm)
    w_agg_mt, p_agg_mt = _agg_wilcoxon(all_diffs_mt)
    
    return {
        "modality": modality_name,
        "n_models": n,
        "model_results": model_results,
        "n_monotonic": n_monotonic,
        # steady vs transient
        "all_diffs": all_diffs,
        "all_cohens_d": all_d,
        "mean_diff": float(np.mean(all_diffs)),
        "std_diff": float(np.std(all_diffs, ddof=1)) if n > 1 else 0.0,
        "agg_w": w_agg,
        "agg_p": p_agg,
        # steady vs mid
        "mean_diff_sm": float(np.mean(all_diffs_sm)),
        "agg_w_sm": w_agg_sm,
        "agg_p_sm": p_agg_sm,
        # mid vs transient
        "mean_diff_mt": float(np.mean(all_diffs_mt)),
        "agg_w_mt": w_agg_mt,
        "agg_p_mt": p_agg_mt,
        # steady vs random
        "all_diffs_sr": all_diffs_sr,
        "all_cohens_d_sr": all_d_sr,
        "mean_diff_sr": float(np.mean(all_diffs_sr)),
        "std_diff_sr": float(np.std(all_diffs_sr, ddof=1)) if n > 1 else 0.0,
        "agg_w_sr": w_agg_sr,
        "agg_p_sr": p_agg_sr,
    }


# ═══════════════════════════════════════════════════════════════
#  输出表格
# ═══════════════════════════════════════════════════════════════

def print_table1(modality_summaries, centers):
    """Table 1: Per-Model Information Retention (with random baseline)"""
    print("\n" + "=" * 200)
    print("Table 1: Information Retention — Cosine Similarity (Full vs Context-Ablated) by DMD Center")
    print("  Higher similarity = mode is more robust to context destruction = encodes global structure")
    print("  Random = randomly selected mode subset (controls for tautology)")
    print("=" * 200)
    
    for ms in modality_summaries:
        mod = ms["modality"]
        print(f"\n── {mod} ({ms['n_models']} models) ──")
        
        c_headers = " | ".join([f"c={c:<4}" for c in centers])
        header = (f"  {'Model':<35} | {'N':>5} | {c_headers} | {'Random':>7} | "
                  f"{'Δ(S-T)':>8} | {'p(S>T)':>12} | {'Δ(S-R)':>8} | {'p(S>R)':>12} | {'Sig':>6}")
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        
        for r in ms["model_results"]:
            cm = r["center_means"]
            c_vals = " | ".join([f"{cm[c]:.4f}" for c in centers])
            
            sig_st = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "n.s."
            sig_sr = "***" if r["p_value_sr"] < 0.001 else "**" if r["p_value_sr"] < 0.01 else "*" if r["p_value_sr"] < 0.05 else "n.s."
            combined_sig = f"{sig_st}/{sig_sr}"
            
            print(f"  {r['model']:<35} | {r['n_stimuli']:>5} | {c_vals} | {r['random_sim_mean']:>7.4f} | "
                  f"{r['mean_diff']:>+8.4f} | {format_p(r['p_value']):>12} | "
                  f"{r['mean_diff_sr']:>+8.4f} | {format_p(r['p_value_sr']):>12} | {combined_sig:>6}")
    
    print("=" * 200)


def print_table2(modality_summaries):
    """Table 2: Modality-Level Summary"""
    print("\n" + "=" * 155)
    print("Table 2: Modality-Level Aggregate — Pairwise Comparisons Across Dynamical Regimes")
    print("  All tests one-sided Wilcoxon on model-level Δ  |  S=Steady(c=1.0) M=Mid(c≈0.5) T=Transient(c=0.0) R=Random")
    print("=" * 155)
    
    header = f"  {'Modality':<12} | {'Test':<8} | {'#Mod':>4} | {'Mean Δ':>8} | {'W-stat':>8} | {'p-value':>12} | {'Conclusion':<40} | {'Mono':>6}"
    print(header)
    print(f"  {'-' * 145}")
    
    for ms in modality_summaries:
        mono_str = f"{ms['n_monotonic']}/{ms['n_models']}"
        tests = [
            ("S > T", ms["mean_diff"],    ms["agg_w"],    ms["agg_p"]),
            ("S > M", ms["mean_diff_sm"], ms["agg_w_sm"], ms["agg_p_sm"]),
            ("M > T", ms["mean_diff_mt"], ms["agg_w_mt"], ms["agg_p_mt"]),
            ("S > R", ms["mean_diff_sr"], ms["agg_w_sr"], ms["agg_p_sr"]),
        ]
        
        for i, (label, md, w, p) in enumerate(tests):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            conclusion = f"{label} {sig}"
            mod_name = ms['modality'] if i == 0 else ""
            mono_col = mono_str if i == 0 else ""
            print(f"  {mod_name:<12} | {label:<8} | {ms['n_models']:>4} | {md:>+8.4f} | {w:>8.1f} | {format_p(p):>12} | {conclusion:<40} | {mono_col:>6}")
        
        print(f"  {'-' * 145}")
    
    print("  Mono = models showing monotonic pattern sim(T) < sim(M) < sim(S)")
    print("=" * 155)


def print_table3(modality_summaries):
    """Table 3: Cross-Modal Comparison"""
    if len(modality_summaries) < 2:
        return
    
    print("\n" + "=" * 100)
    print("Table 3: Cross-Modal — Does the Steady-State Robustness Advantage Differ Across Modalities?")
    print("=" * 100)
    
    if len(modality_summaries) >= 3:
        all_diffs = [ms["all_diffs"] for ms in modality_summaries]
        h_stat, kw_p = kruskal(*all_diffs)
        print(f"  Kruskal-Wallis H = {h_stat:.4f}, p = {format_p(kw_p)}")
    
    print(f"\n  {'Comparison':<25} | {'U-stat':>10} | {'p-value':>12} | {'Direction':<40}")
    print(f"  {'-' * 95}")
    
    for i in range(len(modality_summaries)):
        for j in range(i + 1, len(modality_summaries)):
            ms1, ms2 = modality_summaries[i], modality_summaries[j]
            u_stat, mw_p = mannwhitneyu(ms1["all_diffs"], ms2["all_diffs"], alternative='two-sided')
            
            if np.mean(ms1["all_diffs"]) > np.mean(ms2["all_diffs"]):
                direction = f"{ms1['modality']} has larger robustness gap"
            else:
                direction = f"{ms2['modality']} has larger robustness gap"
            
            sig = "***" if mw_p < 0.001 else "**" if mw_p < 0.01 else "*" if mw_p < 0.05 else "n.s."
            label = f"{ms1['modality']} vs {ms2['modality']}"
            print(f"  {label:<25} | {u_stat:>10.1f} | {format_p(mw_p):>12} | {direction} ({sig})")
    
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════════════════════════

def plot_summary(modality_summaries, centers, save_path="info_retention_summary.png"):
    """跨模态汇总图"""
    n_mod = len(modality_summaries)
    fig, axes = plt.subplots(1, n_mod, figsize=(5 * n_mod, 5), sharey=True)
    if n_mod == 1:
        axes = [axes]
    
    colors_map = {"Vision": "#2563eb", "Audio": "#7c3aed", "Language": "#ea580c"}
    
    for ax, ms in zip(axes, modality_summaries):
        mod = ms["modality"]
        color = colors_map.get(mod, "#333333")
        
        # 每个模型一条线：center → mean similarity
        for r in ms["model_results"]:
            cm = r["center_means"]
            y_vals = [cm[c] for c in centers]
            ax.plot(centers, y_vals, 'o-', alpha=0.4, lw=1, ms=3, color=color)
        
        # 模态均值
        mean_by_center = []
        for c in centers:
            vals = [r["center_means"][c] for r in ms["model_results"]]
            mean_by_center.append(np.mean(vals))
        ax.plot(centers, mean_by_center, 's-', color=color, lw=2.5, ms=8, 
                label=f"Mean (n={ms['n_models']})", zorder=10)
        
        ax.set_xlabel("DMD Center (spectral radius)", fontsize=11)
        ax.set_title(f"{mod}\nΔ={ms['mean_diff']:+.4f}, p={format_p(ms['agg_p'])}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, max(centers) + 0.1)
    
    axes[0].set_ylabel("Cosine Similarity\n(full vs context-ablated)", fontsize=11)
    fig.suptitle("Information Retention Across Dynamical Modes\n"
                 "Higher = more robust to context destruction = encodes global structure",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Plot] → {save_path}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Information Retention Analysis")
    parser.add_argument("--device",       type=str,   default="cuda")
    parser.add_argument("--keep_ratio",   type=float, default=0.5)
    parser.add_argument("--sigma",        type=float, default=0.1)
    parser.add_argument("--dmd_k",        type=int,   default=3)
    parser.add_argument("--max_img",      type=int,   default=500)
    parser.add_argument("--max_audio",    type=int,   default=50)
    parser.add_argument("--save_root",    type=str,   default="results/info_retention")
    
    # 数据路径
    parser.add_argument("--img_root",     type=str,   default="data/image_data/images")
    parser.add_argument("--audio_dir",    type=str,   default="data/audio_data/ds003020-download/stimuli")
    parser.add_argument("--lang_csv",     type=str,   default="data/language_data/EN/lppEN_word_information.csv")
    
    # 选择跑哪些模态
    parser.add_argument("--modalities",   nargs='+',  default=["vision", "audio", "language"],
                        choices=["vision", "audio", "language"])
    args = parser.parse_args()
    
    os.makedirs(args.save_root, exist_ok=True)
    
    centers = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # ── 模型配置 ──
    VISION_MODELS = [
        "facebook/dino-vitb16",
        "facebook/dinov2-base",
        "facebook/dinov2-large",
        "google/vit-base-patch16-224-in21k",
        "facebook/vit-mae-base",
        "facebook/vit-msn-base",
    ]
    
    AUDIO_MODELS = [
        "facebook/wav2vec2-base-960h",
        "facebook/hubert-base-ls960",
        "microsoft/wavlm-base",
        "facebook/data2vec-audio-base",
    ]
    
    LANGUAGE_MODELS = [
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "albert-base-v2",
    ]
    
    all_modality_summaries = []
    
    # ════════════════════════════════════════════════════════════
    #  Vision
    # ════════════════════════════════════════════════════════════
    if "vision" in args.modalities:
        print(f"\n{'='*60}\n  VISION\n{'='*60}")
        vision_results = []
        for model_name in VISION_MODELS:
            print(f"\n  [{model_name}]")
            try:
                X_full, X_mask = collect_vision_data(
                    model_name, args.img_root, device=args.device,
                    max_samples=args.max_img, keep_ratio=args.keep_ratio)
                result = analyze_one_model(X_full, X_mask, model_name, centers,
                                           k=args.dmd_k, sigma=args.sigma)
                vision_results.append(result)
            except Exception as e:
                print(f"    ⚠️ 跳过 {model_name}: {e}")
        
        if vision_results:
            ms = aggregate_modality(vision_results, "Vision")
            all_modality_summaries.append(ms)
    
    # ════════════════════════════════════════════════════════════
    #  Audio
    # ════════════════════════════════════════════════════════════
    if "audio" in args.modalities:
        print(f"\n{'='*60}\n  AUDIO\n{'='*60}")
        audio_results = []
        for model_name in AUDIO_MODELS:
            print(f"\n  [{model_name}]")
            try:
                X_full, X_mask = collect_audio_data(
                    model_name, args.audio_dir, device=args.device,
                    max_samples=args.max_audio, keep_ratio=args.keep_ratio)
                result = analyze_one_model(X_full, X_mask, model_name, centers,
                                           k=args.dmd_k, sigma=args.sigma)
                audio_results.append(result)
            except Exception as e:
                print(f"    ⚠️ 跳过 {model_name}: {e}")
        
        if audio_results:
            ms = aggregate_modality(audio_results, "Audio")
            all_modality_summaries.append(ms)
    
    # ════════════════════════════════════════════════════════════
    #  Language
    # ════════════════════════════════════════════════════════════
    if "language" in args.modalities:
        print(f"\n{'='*60}\n  LANGUAGE\n{'='*60}")
        lang_results = []
        for model_name in LANGUAGE_MODELS:
            print(f"\n  [{model_name}]")
            try:
                X_full, X_mask = collect_language_data(
                    model_name, args.lang_csv, device=args.device,
                    keep_ratio=args.keep_ratio)
                result = analyze_one_model(X_full, X_mask, model_name, centers,
                                           k=args.dmd_k, sigma=args.sigma)
                lang_results.append(result)
            except Exception as e:
                print(f"    ⚠️ 跳过 {model_name}: {e}")
        
        if lang_results:
            ms = aggregate_modality(lang_results, "Language")
            all_modality_summaries.append(ms)
    
    # ════════════════════════════════════════════════════════════
    #  输出
    # ════════════════════════════════════════════════════════════
    if not all_modality_summaries:
        print("\n⚠️ 没有有效结果")
        return
    
    print_table1(all_modality_summaries, centers)
    print_table2(all_modality_summaries)
    print_table3(all_modality_summaries)
    
    plot_summary(all_modality_summaries, centers,
                 save_path=os.path.join(args.save_root, "info_retention_summary.png"))
    
    # ── 保存原始数据 ──
    save_data = {}
    for ms in all_modality_summaries:
        key = ms["modality"].lower()
        save_data[f"{key}_diffs"] = ms["all_diffs"]
        save_data[f"{key}_cohens_d"] = ms["all_cohens_d"]
        for r in ms["model_results"]:
            mkey = r["model"].replace("/", "_")
            for c in centers:
                save_data[f"{key}_{mkey}_c{c}"] = r["per_center_sims"][c]
    
    npz_path = os.path.join(args.save_root, "info_retention_raw.npz")
    np.savez(npz_path, **save_data)
    print(f"\n原始数据已保存: {npz_path}")


if __name__ == "__main__":
    main()