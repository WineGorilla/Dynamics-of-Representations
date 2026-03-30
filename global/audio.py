# """
# 跨模态上下文截断消融实验 —— 语音模型
# =========================================
# 实验逻辑:
#   1. 对同一批音频，分别提取「完整音频」和「截断音频」的 per-layer embeddings
#   2. 对每个 sample 的 (L, d) 矩阵做 soft DMD fusion → 得到 (d,) 向量
#   3. 计算 full vs truncated 的 cosine 相似度
#   4. 不同 dmd_center 下对比（瞬态 vs 稳态）

# 用法:
#   python ablation_audio_crop.py \
#       --model_name facebook/wav2vec2-base-960h \
#       --audio_dir  data/audio_data/ds003020-download/stimuli \
#       --save_root  results/ablation_audio \
#       --trunc_ratio 0.5 \
#       --dmd_center 1.0 \
#       --device mps
# """

# import argparse
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import numpy as np
# import matplotlib.pyplot as plt
# import librosa
# import torch

# from glob import glob
# from tqdm import tqdm

# from utils.dmd import fuse_layers_single_soft_dmd
# from utils.encoder.audio_encoder import load_audio_model


# # ─────────────────────────────────────────────────────────────────────────────
# # 1. 音频加载与截断
# # ─────────────────────────────────────────────────────────────────────────────

# def load_audio_np(file_path: str, sr: int = 16000) -> np.ndarray:
#     y, _ = librosa.load(file_path, sr=sr, mono=True)
#     return y


# def random_frame_keep_audio(y: np.ndarray, ratio: float, frame_sec: float = 0.02, sr: int = 16000) -> np.ndarray:
#     """
#     随机保留 ratio 比例的时间帧，其余帧置零。
#     和视觉 patch keep、语言词 keep 对应。
#     """
#     frame_size = max(1, int(sr * frame_sec))
#     n_frames = len(y) // frame_size
#     n_keep = max(1, int(n_frames * ratio))

#     keep_idx = set(np.random.choice(n_frames, size=n_keep, replace=False))
#     y_out = np.zeros_like(y)
#     for idx in keep_idx:
#         start = idx * frame_size
#         y_out[start:start + frame_size] = y[start:start + frame_size]
#     return y_out


# # ─────────────────────────────────────────────────────────────────────────────
# # 2. 单段音频 → per-layer embedding（均值池化）
# # ─────────────────────────────────────────────────────────────────────────────

# def get_embedding_from_array(y: np.ndarray, processor, model, device: str, sr: int = 16000, tr: float = 2.0) -> np.ndarray:
#     """
#     输入: 1D numpy waveform
#     输出: (L, d)  —— 按 tr 分块处理，每块取均值池化，最后对所有块取均值
#     复用已有 get_audio_embeddings 的分块逻辑。
#     """
#     chunk_size = int(sr * tr)
#     chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]
#     n_layers = model.config.num_hidden_layers + 1
#     layer_accum = [[] for _ in range(n_layers)]

#     for chunk in chunks:
#         if len(chunk) < chunk_size:
#             chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
#         inputs = processor(chunk, sampling_rate=sr, return_tensors="pt").to(device)
#         with torch.no_grad():
#             outputs = model(**inputs, output_hidden_states=True)
#             hidden_states = outputs.hidden_states
#         for l, h in enumerate(hidden_states):
#             emb = h.mean(dim=1).squeeze(0).cpu().numpy()
#             layer_accum[l].append(emb)

#     # (L, T, d) → 对 T 取均值 → (L, d)
#     X = np.stack([np.stack(layer_accum[l], axis=0).mean(axis=0) for l in range(n_layers)], axis=0)
#     return X.astype(np.float32)


# # ─────────────────────────────────────────────────────────────────────────────
# # 3. DMD Fusion
# # ─────────────────────────────────────────────────────────────────────────────

# def dmd_fuse_samples(X_LNd: np.ndarray, k: int = 3, center: float = 1.0, sigma: float = 0.1) -> np.ndarray:
#     """
#     输入: (L, N, d)
#     输出: (N, d)
#     """
#     L, N, d = X_LNd.shape
#     fused = np.zeros((N, d), dtype=np.float32)
#     for n in tqdm(range(N), desc="DMD fusion"):
#         fused[n] = fuse_layers_single_soft_dmd(X_LNd[:, n, :], r=k, center=center, sigma=sigma)
#     return fused


# # ─────────────────────────────────────────────────────────────────────────────
# # 4. Cosine 相似度
# # ─────────────────────────────────────────────────────────────────────────────

# def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
#     """A, B: (N, d) → per-sample cosine sim (N,)"""
#     A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
#     B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
#     return (A_norm * B_norm).sum(axis=1)


# def cosine_per_layer(X_full: np.ndarray, X_trunc: np.ndarray) -> np.ndarray:
#     """X_full, X_trunc: (L, N, d) → per-layer mean cosine sim (L,)"""
#     return np.array([
#         cosine_sim(X_full[l], X_trunc[l]).mean()
#         for l in range(X_full.shape[0])
#     ])


# # ─────────────────────────────────────────────────────────────────────────────
# # 5. 可视化
# # ─────────────────────────────────────────────────────────────────────────────

# def plot_results(layer_sims, dmd_sim_mean, dmd_sim_std, model_name, trunc_ratio, dmd_center, save_path):
#     L = len(layer_sims)
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
#     fig.suptitle(
#         f"Audio Context Ablation — {model_name}\n"
#         f"(random frame keep ratio={trunc_ratio}, dmd_center={dmd_center})",
#         fontsize=13, fontweight='bold', y=1.01
#     )

#     ax = axes[0]
#     ax.plot(range(L), layer_sims, 'o-', color='#7c3aed', lw=2, ms=5)
#     ax.axhline(y=layer_sims.mean(), color='gray', ls='--', lw=1.2,
#                label=f'mean={layer_sims.mean():.3f}')
#     ax.set_xlabel("Layer index", fontsize=11)
#     ax.set_ylabel("Cosine similarity (full vs truncated)", fontsize=11)
#     ax.set_title("Per-layer similarity", fontsize=12)
#     ax.set_ylim(0, 1.05)
#     ax.legend(fontsize=9)
#     ax.grid(True, alpha=0.3)

#     ax2 = axes[1]
#     ax2.bar(['DMD-fused'], [dmd_sim_mean], yerr=[dmd_sim_std],
#             color='#d97706', alpha=0.85, capsize=8, width=0.4,
#             error_kw={'lw': 2})
#     ax2.set_ylim(0, 1.05)
#     ax2.set_ylabel("Cosine similarity", fontsize=11)
#     ax2.set_title(f"After DMD fusion (center={dmd_center})\n"
#                   f"mean={dmd_sim_mean:.3f} ± {dmd_sim_std:.3f}", fontsize=12)
#     ax2.grid(True, alpha=0.3, axis='y')

#     verdict = (
#         "⚠ High sim → this mode insensitive to truncation"
#         if dmd_sim_mean > 0.85
#         else "✓ Low sim → this mode captures global context"
#     )
#     fig.text(0.5, -0.04, verdict, ha='center', fontsize=11,
#              color='#dc2626' if dmd_sim_mean > 0.85 else '#15803d',
#              fontweight='bold')

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"[Plot] → {save_path}")


# # ─────────────────────────────────────────────────────────────────────────────
# # 6. 主流程
# # ─────────────────────────────────────────────────────────────────────────────

# def run_ablation(
#     model_name, audio_dir, save_root,
#     trunc_ratio=0.75, dmd_k=3, dmd_sigma=0.1,
#     device="mps", sr=16000, max_samples=50,
# ):
#     os.makedirs(save_root, exist_ok=True)
#     model_tag = model_name.split("/")[-1]

#     # ── 收集音频路径 ──────────────────────────────────────────────────────────
#     print("[1/5] Collecting audio files ...")
#     audio_paths = sorted(glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))
#     audio_paths += sorted(glob(os.path.join(audio_dir, "**", "*.mp3"), recursive=True))
#     audio_paths = audio_paths[:max_samples]
#     print(f"   → {len(audio_paths)} audio files")
#     if not audio_paths:
#         raise ValueError(f"No audio files found in {audio_dir}")

#     # ── 加载模型 ──────────────────────────────────────────────────────────────
#     print("[2/5] Loading model ...")
#     processor, model = load_audio_model(model_name, device=device)

#     # ── 提取 embeddings ───────────────────────────────────────────────────────
#     print("[3/5] Extracting embeddings ...")
#     full_embeds  = []
#     trunc_embeds = []

#     for path in tqdm(audio_paths, desc="Encoding"):
#         y = load_audio_np(path, sr=sr)
#         y_trunc = random_frame_keep_audio(y, trunc_ratio, sr=sr)

#         emb_full  = get_embedding_from_array(y,       processor, model, device, sr)  # (L, d)
#         emb_trunc = get_embedding_from_array(y_trunc, processor, model, device, sr)  # (L, d)

#         full_embeds.append(emb_full)
#         trunc_embeds.append(emb_trunc)

#     # stack → (L, N, d)
#     X_full  = np.stack(full_embeds,  axis=1).astype(np.float32)  # (L, N, d)
#     X_trunc = np.stack(trunc_embeds, axis=1).astype(np.float32)
#     print(f"   shape: {X_full.shape}  (L, N, d)")

#     # ── Per-layer cosine sim ──────────────────────────────────────────────────
#     print("[4/5] Per-layer cosine similarity ...")
#     layer_sims = cosine_per_layer(X_full, X_trunc)
#     for l, s in enumerate(layer_sims):
#         print(f"   Layer {l:02d}: {s:.4f}")

#     # ── DMD Fusion：同时跑所有 center 值 ────────────────────────────────────
#     print("[5/5] No-DMD baseline + DMD fusion across all centers ...")

#     # ── 无DMD基线：所有层均值池化 ────────────────────────────────────────────
#     X_full_mean  = X_full.mean(axis=0)        # (N, d)
#     X_other_mean = X_trunc.mean(axis=0) # (N, d)
#     sim_nodmd = cosine_sim(X_full_mean, X_other_mean)
#     nodmd_mean = float(sim_nodmd.mean())
#     nodmd_std  = float(sim_nodmd.std())
#     print(f"   no_dmd (mean pool): {nodmd_mean:.4f} ± {nodmd_std:.4f}")

#     centers = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
#     results_by_center = {}

#     for c in centers:
#         fused_full  = dmd_fuse_samples(X_full,  k=dmd_k, center=c, sigma=dmd_sigma)
#         fused_trunc = dmd_fuse_samples(X_trunc, k=dmd_k, center=c, sigma=dmd_sigma)
#         per_sample_sim = cosine_sim(fused_full, fused_trunc)
#         sim_mean = float(per_sample_sim.mean())
#         sim_std  = float(per_sample_sim.std())
#         results_by_center[c] = {"mean": sim_mean, "std": sim_std,
#                                  "fused_full": fused_full, "fused_trunc": fused_trunc}
#         print(f"   center={c:.1f}: {sim_mean:.4f} ± {sim_std:.4f}")

#     # ── 保存 ──────────────────────────────────────────────────────────────────
#     tag = f"{model_tag}_trunc{trunc_ratio}"
#     npz_path = os.path.join(save_root, f"{tag}.npz")
#     save_dict = {"layer_sims": layer_sims}
#     for c, r in results_by_center.items():
#         save_dict[f"sim_mean_c{c}"] = np.array([r["mean"]])
#         save_dict[f"sim_std_c{c}"]  = np.array([r["std"]])
#     np.savez(npz_path, **save_dict)
#     print(f"   Saved → {npz_path}")

#     # ── 对比图：所有 center 的 DMD sim ───────────────────────────────────────
#     fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
#     fig.suptitle(f"Audio Context Ablation — {model_tag}\n(trunc_ratio={trunc_ratio})",
#                  fontsize=13, fontweight="bold", y=1.01)

#     ax = axes[0]
#     ax.plot(range(len(layer_sims)), layer_sims, "o-", color="#7c3aed", lw=2, ms=5)
#     ax.axhline(y=layer_sims.mean(), color="gray", ls="--", lw=1.2,
#                label=f"mean={layer_sims.mean():.3f}")
#     ax.set_xlabel("Layer index"); ax.set_ylabel("Cosine similarity (full vs masked)")
#     ax.set_title("Per-layer similarity"); ax.set_ylim(0, 1.05)
#     ax.legend(); ax.grid(True, alpha=0.3)

#     ax2 = axes[1]
#     c_vals  = list(results_by_center.keys())
#     means   = [results_by_center[c]["mean"] for c in c_vals]
#     stds    = [results_by_center[c]["std"]  for c in c_vals]
#     colors  = ["#ef4444" if m < 0 else "#3b82f6" for m in means]
#     ax2.bar([str(c) for c in c_vals], means, yerr=stds, color=colors,
#             alpha=0.8, capsize=6, error_kw={"lw": 1.5})
#     ax2.axhline(y=0, color="black", lw=0.8)
#     ax2.set_xlabel("DMD center (spectral radius)"); ax2.set_ylabel("Cosine similarity")
#     ax2.set_title("DMD-fused similarity vs center"); ax2.grid(True, alpha=0.3, axis="y")

#     plt.tight_layout()
#     plot_path = os.path.join(save_root, f"{tag}_allcenters_plot.png")
#     plt.savefig(plot_path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"   [Plot] → {plot_path}")




#     # ── 横向汇总表格 ─────────────────────────────────────────────────────────
#     _cw   = 9
#     _cols = ["No_DMD"] + ["c=" + str(c) for c in centers]
#     _vals = [nodmd_mean] + [results_by_center[c]["mean"] for c in centers]
#     _hdr  = "  " + f"{'Model':<30}" + "".join(f"{h:>{_cw}}" for h in _cols)
#     _row  = "  " + f"{model_tag:<30}" + "".join(f"{v:>{_cw}.4f}" for v in _vals)
#     _sep  = "-" * len(_hdr)
#     print(f"\nAudio Ablation  ratio={trunc_ratio}")
#     print(_sep); print(_hdr); print(_sep); print(_row); print(_sep + "\n")

#     return {
#         "model": model_name,
#         "trunc_ratio": trunc_ratio,
#         "n_samples": len(audio_paths),
#         "n_layers": int(X_full.shape[0]),
#         "layer_sims": layer_sims.tolist(),
#         "nodmd_mean": nodmd_mean,
#         "results_by_center": {str(c): r["mean"] for c, r in results_by_center.items()},
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # CLI
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name",  type=str,   default="facebook/wav2vec2-base-960h")
#     parser.add_argument("--audio_dir",   type=str,   default="data/audio_data/ds003020-download/stimuli")
#     parser.add_argument("--save_root",   type=str,   default="results/ablation_audio")
#     parser.add_argument("--trunc_ratio", type=float, default=0.5,
#                         help="保留前多少比例的音频，0.5=只保留前半段")
#     parser.add_argument("--dmd_k",       type=int,   default=3)
#     parser.add_argument("--dmd_sigma",   type=float, default=0.1)
#     parser.add_argument("--device",      type=str,   default="mps")
#     parser.add_argument("--sr",          type=int,   default=16000)
#     parser.add_argument("--max_samples", type=int,   default=50)
#     parser.add_argument("--all_models",  action="store_true")
#     args = parser.parse_args()

#     ALL_AUDIO_MODELS = [
#         "facebook/wav2vec2-base-960h",
#         "facebook/hubert-base-ls960",
#         "microsoft/wavlm-base",
#         "facebook/data2vec-audio-base",
#     ]
#     models_to_run = ALL_AUDIO_MODELS if args.all_models else [args.model_name]

#     all_results = []
#     for mn in models_to_run:
#         print(f"\n{'='*60}\n  Model: {mn}\n{'='*60}")
#         res = run_ablation(
#             model_name=mn,
#             audio_dir=args.audio_dir,
#             save_root=args.save_root,
#             trunc_ratio=args.trunc_ratio,
#             dmd_k=args.dmd_k,
#             dmd_sigma=args.dmd_sigma,
#             device=args.device,
#             sr=args.sr,
#             max_samples=args.max_samples,
#         )
#         all_results.append(res)

#     print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
#     print(f"{'Model':<35} {'center':>8} {'DMD sim':>10}  {'std':>8}  Interpretation")
#     print("-" * 78)
#     for r in all_results:
#         tag = r["model"].split("/")[-1]
#         verdict = "HIGH → insensitive" if r["dmd_sim_mean"] > 0.85 else "LOW  → context-sensitive"
#         print(f"{tag:<35} {r['dmd_center']:>8.1f} {r['dmd_sim_mean']:>10.4f}  {r['dmd_sim_std']:>8.4f}  {verdict}")



"""
跨模态上下文截断消融实验 —— 视觉模型
=========================================
实验逻辑:
  1. 对同一批图像，分别提取「完整图」和「中心裁剪图」的 per-layer embeddings
  2. 对每个 sample 的 (L, d) 矩阵做 DMD fusion → 得到 (d,) 向量
  3. 计算 full vs cropped 的 cosine 相似度
  4. 统计并可视化结果

用法:
  python ablation_vision_crop.py \
      --model_name facebook/dinov2-base \
      --img_root   data/image_data/images \
      --events_dir data/image_data/ds004192-download \
      --save_root  results/ablation_vision \
      --crop_ratio 0.5 \
      --device     mps
"""


"""
跨模态上下文截断消融实验 —— 语音模型
=========================================
实验逻辑:
  1. 对同一批音频，分别提取「完整音频」和「截断音频」的 per-layer embeddings
  2. 对每个 sample 的 (L, d) 矩阵做 soft DMD fusion → 得到 (d,) 向量
  3. 计算 full vs truncated 的 cosine 相似度
  4. 不同 dmd_center 下对比（瞬态 vs 稳态）

用法:
  python ablation_audio_crop.py \
      --model_name facebook/wav2vec2-base-960h \
      --audio_dir  data/audio_data/ds003020-download/stimuli \
      --save_root  results/ablation_audio \
      --trunc_ratio 0.5 \
      --dmd_center 1.0 \
      --device mps
"""
"""
跨模态上下文截断消融实验 —— 语音模型
=========================================
实验逻辑:
  1. 对同一批音频，分别提取「完整音频」和「截断音频」的 per-layer embeddings
  2. 对每个 sample 的 (L, d) 矩阵做 soft DMD fusion → 得到 (d,) 向量
  3. 计算 full vs truncated 的 cosine 相似度
  4. 不同 dmd_center 下对比（瞬态 vs 稳态）

用法:
  python ablation_audio_crop.py \
      --model_name facebook/wav2vec2-base-960h \
      --audio_dir  data/audio_data/ds003020-download/stimuli \
      --save_root  results/ablation_audio \
      --trunc_ratio 0.5 \
      --dmd_center 1.0 \
      --device mps
"""

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch

from glob import glob
from tqdm import tqdm

from core.dmd import fuse_layers_single_soft_dmd
from core.encoder.audio_encoder import load_audio_model


# ─────────────────────────────────────────────────────────────────────────────
# 1. 音频加载与截断
# ─────────────────────────────────────────────────────────────────────────────

def load_audio_np(file_path: str, sr: int = 16000) -> np.ndarray:
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return y


def random_frame_keep_audio(y: np.ndarray, ratio: float, frame_sec: float = 0.02, sr: int = 16000) -> np.ndarray:
    """
    随机保留 ratio 比例的时间帧，其余帧置零。
    和视觉 patch keep、语言词 keep 对应。
    """
    frame_size = max(1, int(sr * frame_sec))
    n_frames = len(y) // frame_size
    n_keep = max(1, int(n_frames * ratio))

    keep_idx = set(np.random.choice(n_frames, size=n_keep, replace=False))
    y_out = np.zeros_like(y)
    for idx in keep_idx:
        start = idx * frame_size
        y_out[start:start + frame_size] = y[start:start + frame_size]
    return y_out


# ─────────────────────────────────────────────────────────────────────────────
# 2. 单段音频 → per-layer embedding（均值池化）
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding_from_array(y: np.ndarray, processor, model, device: str, sr: int = 16000, tr: float = 2.0) -> np.ndarray:
    """
    输入: 1D numpy waveform
    输出: (L, d)  —— 按 tr 分块处理，每块取均值池化，最后对所有块取均值
    复用已有 get_audio_embeddings 的分块逻辑。
    """
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
            hidden_states = outputs.hidden_states
        for l, h in enumerate(hidden_states):
            emb = h.mean(dim=1).squeeze(0).cpu().numpy()
            layer_accum[l].append(emb)

    # (L, T, d) → 对 T 取均值 → (L, d)
    X = np.stack([np.stack(layer_accum[l], axis=0).mean(axis=0) for l in range(n_layers)], axis=0)
    return X.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DMD Fusion
# ─────────────────────────────────────────────────────────────────────────────

def dmd_fuse_samples(X_LNd: np.ndarray, k: int = 3, center: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    """
    输入: (L, N, d)
    输出: (N, d)
    """
    L, N, d = X_LNd.shape
    fused = np.zeros((N, d), dtype=np.float32)
    for n in tqdm(range(N), desc="DMD fusion"):
        fused[n] = fuse_layers_single_soft_dmd(X_LNd[:, n, :], r=k, center=center, sigma=sigma)
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cosine 相似度
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A, B: (N, d) → per-sample cosine sim (N,)"""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (A_norm * B_norm).sum(axis=1)


def cosine_per_layer(X_full: np.ndarray, X_trunc: np.ndarray) -> np.ndarray:
    """X_full, X_trunc: (L, N, d) → per-layer mean cosine sim (L,)"""
    return np.array([
        cosine_sim(X_full[l], X_trunc[l]).mean()
        for l in range(X_full.shape[0])
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 5. 可视化
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(layer_sims, dmd_sim_mean, dmd_sim_std, model_name, trunc_ratio, dmd_center, save_path):
    L = len(layer_sims)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Audio Context Ablation — {model_name}\n"
        f"(random frame keep ratio={trunc_ratio}, dmd_center={dmd_center})",
        fontsize=13, fontweight='bold', y=1.01
    )

    ax = axes[0]
    ax.plot(range(L), layer_sims, 'o-', color='#7c3aed', lw=2, ms=5)
    ax.axhline(y=layer_sims.mean(), color='gray', ls='--', lw=1.2,
               label=f'mean={layer_sims.mean():.3f}')
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Cosine similarity (full vs truncated)", fontsize=11)
    ax.set_title("Per-layer similarity", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(['DMD-fused'], [dmd_sim_mean], yerr=[dmd_sim_std],
            color='#d97706', alpha=0.85, capsize=8, width=0.4,
            error_kw={'lw': 2})
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cosine similarity", fontsize=11)
    ax2.set_title(f"After DMD fusion (center={dmd_center})\n"
                  f"mean={dmd_sim_mean:.3f} ± {dmd_sim_std:.3f}", fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    verdict = (
        "⚠ High sim → this mode insensitive to truncation"
        if dmd_sim_mean > 0.85
        else "✓ Low sim → this mode captures global context"
    )
    fig.text(0.5, -0.04, verdict, ha='center', fontsize=11,
             color='#dc2626' if dmd_sim_mean > 0.85 else '#15803d',
             fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    model_name, audio_dir, save_root,
    trunc_ratio=0.75, dmd_k=3, dmd_sigma=0.1,
    device="mps", sr=16000, max_samples=50,
):
    os.makedirs(save_root, exist_ok=True)
    model_tag = model_name.split("/")[-1]

    # ── 收集音频路径 ──────────────────────────────────────────────────────────
    print("[1/5] Collecting audio files ...")
    audio_paths = sorted(glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))
    audio_paths += sorted(glob(os.path.join(audio_dir, "**", "*.mp3"), recursive=True))
    audio_paths = audio_paths[:max_samples]
    print(f"   → {len(audio_paths)} audio files")
    if not audio_paths:
        raise ValueError(f"No audio files found in {audio_dir}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print("[2/5] Loading model ...")
    processor, model = load_audio_model(model_name, device=device)

    # ── 提取 embeddings ───────────────────────────────────────────────────────
    print("[3/5] Extracting embeddings ...")
    full_embeds  = []
    trunc_embeds = []

    for path in tqdm(audio_paths, desc="Encoding"):
        y = load_audio_np(path, sr=sr)
        y_trunc = random_frame_keep_audio(y, trunc_ratio, sr=sr)

        emb_full  = get_embedding_from_array(y,       processor, model, device, sr)  # (L, d)
        emb_trunc = get_embedding_from_array(y_trunc, processor, model, device, sr)  # (L, d)

        full_embeds.append(emb_full)
        trunc_embeds.append(emb_trunc)

    # stack → (L, N, d)
    X_full  = np.stack(full_embeds,  axis=1).astype(np.float32)  # (L, N, d)
    X_trunc = np.stack(trunc_embeds, axis=1).astype(np.float32)
    print(f"   shape: {X_full.shape}  (L, N, d)")

    # ── Per-layer cosine sim ──────────────────────────────────────────────────
    print("[4/5] Per-layer cosine similarity ...")
    layer_sims = cosine_per_layer(X_full, X_trunc)
    for l, s in enumerate(layer_sims):
        print(f"   Layer {l:02d}: {s:.4f}")

    # ── DMD Fusion：同时跑所有 center 值 ────────────────────────────────────
    print("[5/5] No-DMD baseline + DMD fusion across all centers ...")

    # ── 无DMD基线：所有层均值池化 ────────────────────────────────────────────
    X_full_mean  = X_full.mean(axis=0)        # (N, d)
    X_other_mean = X_trunc.mean(axis=0) # (N, d)
    sim_nodmd = cosine_sim(X_full_mean, X_other_mean)
    nodmd_mean = float(sim_nodmd.mean())
    nodmd_std  = float(sim_nodmd.std())
    print(f"   no_dmd (mean pool): {nodmd_mean:.4f} ± {nodmd_std:.4f}")

    centers = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    results_by_center = {}

    for c in centers:
        fused_full  = dmd_fuse_samples(X_full,  k=dmd_k, center=c, sigma=dmd_sigma)
        fused_trunc = dmd_fuse_samples(X_trunc, k=dmd_k, center=c, sigma=dmd_sigma)
        per_sample_sim = cosine_sim(fused_full, fused_trunc)
        sim_mean = float(per_sample_sim.mean())
        sim_std  = float(per_sample_sim.std())
        results_by_center[c] = {"mean": sim_mean, "std": sim_std,
                                 "fused_full": fused_full, "fused_trunc": fused_trunc}
        print(f"   center={c:.1f}: {sim_mean:.4f} ± {sim_std:.4f}")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    tag = f"{model_tag}_trunc{trunc_ratio}"
    npz_path = os.path.join(save_root, f"{tag}.npz")
    save_dict = {"layer_sims": layer_sims}
    for c, r in results_by_center.items():
        save_dict[f"sim_mean_c{c}"] = np.array([r["mean"]])
        save_dict[f"sim_std_c{c}"]  = np.array([r["std"]])
    np.savez(npz_path, **save_dict)
    print(f"   Saved → {npz_path}")

    # ── 对比图：所有 center 的 DMD sim ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"Audio Context Ablation — {model_tag}\n(trunc_ratio={trunc_ratio})",
                 fontsize=13, fontweight="bold", y=1.01)

    ax = axes[0]
    ax.plot(range(len(layer_sims)), layer_sims, "o-", color="#7c3aed", lw=2, ms=5)
    ax.axhline(y=layer_sims.mean(), color="gray", ls="--", lw=1.2,
               label=f"mean={layer_sims.mean():.3f}")
    ax.set_xlabel("Layer index"); ax.set_ylabel("Cosine similarity (full vs masked)")
    ax.set_title("Per-layer similarity"); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    c_vals  = list(results_by_center.keys())
    means   = [results_by_center[c]["mean"] for c in c_vals]
    stds    = [results_by_center[c]["std"]  for c in c_vals]
    colors  = ["#ef4444" if m < 0 else "#3b82f6" for m in means]
    ax2.bar([str(c) for c in c_vals], means, yerr=stds, color=colors,
            alpha=0.8, capsize=6, error_kw={"lw": 1.5})
    ax2.axhline(y=0, color="black", lw=0.8)
    ax2.set_xlabel("DMD center (spectral radius)"); ax2.set_ylabel("Cosine similarity")
    ax2.set_title("DMD-fused similarity vs center"); ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(save_root, f"{tag}_allcenters_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [Plot] → {plot_path}")




    # ── 横向汇总表格 ─────────────────────────────────────────────────────────
    _cw   = 9
    _cols = ["No_DMD"] + ["c=" + str(c) for c in centers]
    _vals = [nodmd_mean] + [results_by_center[c]["mean"] for c in centers]
    _hdr  = "  " + f"{'Model':<30}" + "".join(f"{h:>{_cw}}" for h in _cols)
    _row  = "  " + f"{model_tag:<30}" + "".join(f"{v:>{_cw}.4f}" for v in _vals)
    _sep  = "-" * len(_hdr)
    print(f"\nAudio Ablation  ratio={trunc_ratio}")
    print(_sep); print(_hdr); print(_sep); print(_row); print(_sep + "\n")

    return {
        "model": model_name,
        "trunc_ratio": trunc_ratio,
        "n_samples": len(audio_paths),
        "n_layers": int(X_full.shape[0]),
        "layer_sims": layer_sims.tolist(),
        "nodmd_mean": nodmd_mean,
        "results_by_center": {str(c): r["mean"] for c, r in results_by_center.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str,   default="facebook/wav2vec2-base-960h")
    parser.add_argument("--audio_dir",   type=str,   default="data/audio_data/ds003020-download/stimuli")
    parser.add_argument("--save_root",   type=str,   default="results/ablation_audio")
    parser.add_argument("--trunc_ratio", type=float, default=0.5,
                        help="保留前多少比例的音频，0.5=只保留前半段")
    parser.add_argument("--dmd_k",       type=int,   default=3)
    parser.add_argument("--dmd_sigma",   type=float, default=0.1)
    parser.add_argument("--device",      type=str,   default="mps")
    parser.add_argument("--sr",          type=int,   default=16000)
    parser.add_argument("--max_samples", type=int,   default=50)
    parser.add_argument("--all_models",  action="store_true")
    args = parser.parse_args()

    ALL_AUDIO_MODELS = [
        "facebook/wav2vec2-base-960h",
        "facebook/hubert-base-ls960",
        "microsoft/wavlm-base",
        "facebook/data2vec-audio-base",
    ]
    models_to_run = ALL_AUDIO_MODELS if args.all_models else [args.model_name]

    all_results = []
    for mn in models_to_run:
        print(f"\n{'='*60}\n  Model: {mn}\n{'='*60}")
        res = run_ablation(
            model_name=mn,
            audio_dir=args.audio_dir,
            save_root=args.save_root,
            trunc_ratio=args.trunc_ratio,
            dmd_k=args.dmd_k,
            dmd_sigma=args.dmd_sigma,
            device=args.device,
            sr=args.sr,
            max_samples=args.max_samples,
        )
        all_results.append(res)

    centers = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    _cw   = 9
    _cols = ["No_DMD"] + ["c=" + str(c) for c in centers]
    _hdr  = "  " + f"{'Model':<30}" + "".join(f"{h:>{_cw}}" for h in _cols)
    _sep  = "-" * len(_hdr)
    print(f"\nAudio Ablation Summary")
    print(_sep); print(_hdr); print(_sep)
    for r in all_results:
        tag = r["model"].split("/")[-1]
        _vals = [r["nodmd_mean"]] + [r["results_by_center"][str(c)] for c in centers]
        _row  = "  " + f"{tag:<30}" + "".join(f"{v:>{_cw}.4f}" for v in _vals)
        print(_row)
    print(_sep)