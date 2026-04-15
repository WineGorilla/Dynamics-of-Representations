"""
Per-Frame Information Retention Across Dynamical Modes (Audio)
==============================================================
和可视化代码完全对应：
  1. 整段音频送入模型，提取所有层的 per-frame token: (L, T, D)
  2. 对每个时间帧独立做 fuse_layers_single_soft_dmd (各 center)
  3. cos(fused_frame, last_layer_frame) = 信息保留度
  4. 模型内 Friedman + 模态级 Friedman omnibus + Wilcoxon post-hoc

用法：
  CUDA_VISIBLE_DEVICES=1 python frame_info_retention_audio.py --device cuda
  CUDA_VISIBLE_DEVICES=1 python frame_info_retention_audio.py --device cuda --max_audio 20
"""

import argparse
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import gc
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import librosa
from glob import glob
from tqdm import tqdm
from scipy.stats import friedmanchisquare
from transformers import (
    AutoProcessor, AutoFeatureExtractor, AutoModel,
    WhisperModel, WhisperFeatureExtractor
)

from core.dmd import fuse_layers_single_soft_dmd


# ═══════════════════════════════════════════════════════════════
#  模型加载
# ═══════════════════════════════════════════════════════════════

def load_audio_model(model_name, device):
    model_lower = model_name.lower()

    if "whisper" in model_lower:
        processor = WhisperFeatureExtractor.from_pretrained(model_name)
        full_model = WhisperModel.from_pretrained(model_name, output_hidden_states=True)
        model = full_model.encoder
        model.config = full_model.config
        model.config.model_type = "whisper"
    else:
        try:
            processor = AutoProcessor.from_pretrained(model_name)
        except Exception:
            processor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    model = model.to(device).eval()
    return processor, model


# ═══════════════════════════════════════════════════════════════
#  Per-frame hidden states 提取（和可视化代码一致）
# ═══════════════════════════════════════════════════════════════

def extract_per_frame_tokens(processor, model, audio_path, device, sr=16000, duration=5.0):
    """
    单段音频 → (L, T, D)
    整段送入模型，保留所有时间帧，不做均值池化。
    duration: 只取前 N 秒，避免 OOM。
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=duration)
    model_type = getattr(model.config, "model_type", "")

    if model_type == "whisper":
        inputs = processor(y, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs["input_features"], output_hidden_states=True)
    else:
        inputs = processor(y, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (1, T, D)

    # (L, T, D)
    token_matrix = np.stack([h[0].cpu().numpy() for h in hidden_states], axis=0)
    return token_matrix.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  工具
# ═══════════════════════════════════════════════════════════════

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def format_p(p):
    if p < 0.0001:   return "< 0.0001"
    elif p < 0.001:  return "< 0.001"
    elif p < 0.01:   return "< 0.01"
    elif p < 0.05:   return "< 0.05"
    else:            return f"{p:.4f}"


# ═══════════════════════════════════════════════════════════════
#  核心：处理一段音频的所有 frame
# ═══════════════════════════════════════════════════════════════

def process_frames(frame_tokens, centers, dmd_k=3, sigma=0.1):
    """
    frame_tokens: (L, T, D)
    对每个时间帧独立做 DMD fusion，和可视化代码完全一致。
    """
    L, T, D = frame_tokens.shape
    last_layer = frame_tokens[-1]  # (T, D)

    sims = {c: [] for c in centers}
    for t in range(T):
        trajectory = frame_tokens[:, t, :]  # (L, D)
        for c in centers:
            try:
                fused = fuse_layers_single_soft_dmd(trajectory, r=dmd_k, center=c, sigma=sigma)
                sims[c].append(cosine_sim(fused, last_layer[t]))
            except Exception:
                sims[c].append(0.0)
    return sims


# ═══════════════════════════════════════════════════════════════
#  单个模型
# ═══════════════════════════════════════════════════════════════

def run_one_model(model_name, audio_paths, centers, device="cuda",
                  sigma=0.1, dmd_k=3, sr=16000, duration=5.0):

    processor, model = load_audio_model(model_name, device)
    tag = model_name.split("/")[-1]

    all_sims = {c: [] for c in centers}

    for audio_path in tqdm(audio_paths, desc=f"  {tag}"):
        try:
            frame_tokens = extract_per_frame_tokens(
                processor, model, audio_path, device, sr=sr, duration=duration)
        except Exception:
            continue

        L, T, D = frame_tokens.shape
        if T < 3:
            continue

        sims = process_frames(frame_tokens, centers, dmd_k, sigma)
        for c in centers:
            all_sims[c].extend(sims[c])

    n_total = len(all_sims[centers[0]])
    if n_total < 100:
        print(f"    ⚠️ {tag}: 数据不足 ({n_total})")
        del processor, model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

    # Friedman test
    center_means = {c: float(np.mean(all_sims[c])) for c in centers}
    best_center = max(center_means, key=center_means.get)

    max_n = 50000
    idx = np.random.choice(n_total, size=min(n_total, max_n), replace=False) if n_total > max_n else np.arange(n_total)
    samples = [np.array(all_sims[c])[idx] for c in centers]
    try:
        chi2, p_friedman = friedmanchisquare(*samples)
    except Exception:
        chi2, p_friedman = 0.0, 1.0

    def _sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
    cm_str = " | ".join([f"c={c}:{center_means[c]:.4f}" for c in centers])
    print(f"    {tag:35s} | n={n_total:>6} | {cm_str} | best=c={best_center} | "
          f"χ²={chi2:.1f} p={format_p(p_friedman)}{_sig(p_friedman)}")

    del processor, model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_total": n_total,
        "center_means": center_means,
        "best_center": best_center,
        "friedman_chi2": chi2,
        "friedman_p": p_friedman,
        "sims": {c: np.array(all_sims[c]) for c in centers},
    }


# ═══════════════════════════════════════════════════════════════
#  模态级汇总
# ═══════════════════════════════════════════════════════════════

def aggregate_results(model_results, group_name, centers):
    n = len(model_results)
    n_sig = sum(1 for r in model_results if r["friedman_p"] < 0.05)

    center_vectors = {c: np.array([r["center_means"][c] for r in model_results]) for c in centers}
    center_summary = {c: {"mean": float(np.mean(center_vectors[c])),
                          "std": float(np.std(center_vectors[c], ddof=1)) if n > 1 else 0.0}
                      for c in centers}

    if n >= 3:
        try:
            chi2, p = friedmanchisquare(*[center_vectors[c] for c in centers])
        except Exception:
            chi2, p = 0.0, 1.0
    else:
        chi2, p = 0.0, 1.0

    return {
        "group": group_name,
        "n_models": n,
        "n_significant": n_sig,
        "center_summary": center_summary,
        "friedman_chi2": chi2,
        "friedman_p": p,
        "model_results": model_results,
    }


# ═══════════════════════════════════════════════════════════════
#  表格
# ═══════════════════════════════════════════════════════════════

def print_table1(group_summaries, centers):
    print("\n" + "=" * 150)
    print("Table 1: Per-Frame Information Retention — cos(DMD_fused, last_layer) by Center")
    print("  Each frame ≈ 20ms. Higher = DMD mode captures more of the final representation.")
    print("=" * 150)

    for gs in group_summaries:
        print(f"\n── {gs['group']} ({gs['n_models']} models, {gs['n_significant']}/{gs['n_models']} significant) ──")
        c_hdrs = " | ".join([f"c={c:>4}" for c in centers])
        header = f"  {'Model':<40} | {'N':>6} | {c_hdrs} | {'Best':>5} | {'χ²':>8} | {'p':>10}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        for r in gs["model_results"]:
            cm = r["center_means"]
            c_vals = " | ".join([f"{cm[c]:>6.4f}" for c in centers])
            name = r["model"].split("/")[-1]
            def _s(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
            print(f"  {name:<40} | {r['n_total']:>6} | {c_vals} | c={r['best_center']:<3} | "
                  f"{r['friedman_chi2']:>8.1f} | {format_p(r['friedman_p']):>10} {_s(r['friedman_p'])}")

    print("=" * 150)


def print_table2(group_summaries, centers):
    from scipy.stats import wilcoxon as wilcoxon_test

    print("\n" + "=" * 115)
    print("Table 2: Cross-Model Universality")
    print("  Friedman omnibus test, followed by Wilcoxon signed-rank post-hoc (two-sided).")
    print("=" * 115)

    def _s(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

    for gs in group_summaries:
        n = gs["n_models"]
        print(f"\n  {gs['group']}  (n = {n} models)")
        print(f"  Friedman χ²({len(centers)-1}) = {gs['friedman_chi2']:.2f},  p = {format_p(gs['friedman_p'])} {_s(gs['friedman_p'])}")
        print()

        header = f"  {'Test':<8} | {'#Models':>7} | {'Mean Δ':>8} | {'Cohen d':>8} | {'W-stat':>8} | {'p-value':>12} | {'Sig':>5}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        vals = {c: np.array([r["center_means"][c] for r in gs["model_results"]]) for c in centers}

        pairs = [
            ("S > T", centers[2], centers[0]),
            ("S > M", centers[2], centers[1]),
            ("M > T", centers[1], centers[0]),
        ]

        for label, c_a, c_b in pairs:
            diff = vals[c_a] - vals[c_b]
            mean_d = float(np.mean(diff))
            std_d = float(np.std(diff, ddof=1)) if n > 1 else 1.0
            cohens_d = mean_d / std_d if std_d > 0 else float('inf')
            if n >= 2 and not np.all(diff == 0):
                w, p = wilcoxon_test(diff, alternative='two-sided')
            else:
                w, p = 0.0, 1.0
            d_str = f"{cohens_d:.2f}" if not np.isinf(cohens_d) else "inf"
            print(f"  {label:<8} | {n:>7} | {mean_d:>+8.4f} | {d_str:>8} | {w:>8.1f} | {format_p(p):>12} | {_s(p):>5}")

        print(f"  {'-' * (len(header) - 2)}")

    print("=" * 115)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

AUDIO_MODELS = [
    "facebook/data2vec-audio-base-960h",
    "facebook/data2vec-audio-base",
    "facebook/data2vec-audio-large-960h",
    "facebook/data2vec-audio-large",
    "facebook/hubert-base-ls960",
    "superb/hubert-base-superb-ks",
    "facebook/hubert-large-ls960-ft",
    "facebook/hubert-xlarge-ls960-ft",
    "asapp/sew-d-mid-100k",
    "asapp/sew-d-small-100k",
    "asapp/sew-d-tiny-100k",
    "asapp/sew-mid-100k",
    "asapp/sew-small-100k",
    "asapp/sew-tiny-100k",
    "microsoft/unispeech-large-1500h-cv",
    "microsoft/unispeech-sat-base-plus",
    "microsoft/unispeech-sat-base",
    "microsoft/unispeech-sat-large",
    "facebook/w2v-bert-2.0",
    "facebook/wav2vec2-base-960h",
    "superb/wav2vec2-base-superb-ks",
    "facebook/wav2vec2-base",
    "facebook/wav2vec2-conformer-rel-pos-large",
    "facebook/wav2vec2-conformer-rope-large-960h-ft",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large",
    "facebook/wav2vec2-large-xlsr-53",
    "facebook/wav2vec2-xls-r-1b",
    "facebook/wav2vec2-xls-r-300m",
    "microsoft/wavlm-base-plus",
    "microsoft/wavlm-base",
    "microsoft/wavlm-large",
    "openai/whisper-base",
    "openai/whisper-medium",
    "openai/whisper-small",
    "openai/whisper-tiny",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",      type=str,   default="cuda")
    parser.add_argument("--audio_dir",   type=str,   default="data/audio_data/stimuli")
    parser.add_argument("--sigma",       type=float, default=0.1)
    parser.add_argument("--dmd_k",       type=int,   default=1)
    parser.add_argument("--max_audio",   type=int,   default=50)
    parser.add_argument("--duration",    type=float, default=5.0,
                        help="每段音频只取前 N 秒")
    parser.add_argument("--sr",          type=int,   default=16000)
    parser.add_argument("--save_root",   type=str,   default="results/frame_retention_audio")
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    centers = [0.0, 0.5, 1.0]

    audio_paths = sorted(glob(os.path.join(args.audio_dir, "**", "*.wav"), recursive=True))
    audio_paths += sorted(glob(os.path.join(args.audio_dir, "**", "*.mp3"), recursive=True))
    audio_paths = audio_paths[:args.max_audio]
    print(f"Audio files: {len(audio_paths)}")
    print(f"Centers = {centers}, sigma = {args.sigma}, dmd_k = {args.dmd_k}")
    print(f"Duration per audio = {args.duration}s")

    all_results = []
    for model_name in AUDIO_MODELS:
        print(f"\n  [{model_name}]")
        try:
            r = run_one_model(model_name, audio_paths, centers,
                              device=args.device, sigma=args.sigma,
                              dmd_k=args.dmd_k, sr=args.sr, duration=args.duration)
            if r:
                all_results.append(r)
        except Exception as e:
            print(f"    ⚠️ {model_name}: {e}")

    if not all_results:
        print("⚠️ 没有结果")
        return

    gs = aggregate_results(all_results, "Audio", centers)
    print_table1([gs], centers)
    print_table2([gs], centers)

    save_data = {}
    for r in all_results:
        mkey = r["model"].replace("/", "_")
        for c in centers:
            save_data[f"{mkey}_c{c}"] = r["sims"][c]

    npz_path = os.path.join(args.save_root, "frame_retention_results.npz")
    np.savez(npz_path, **save_data)
    print(f"\n保存: {npz_path}")


if __name__ == "__main__":
    main()