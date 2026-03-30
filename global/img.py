# """
# 跨模态上下文截断消融实验 —— 视觉模型
# =========================================
# 实验逻辑:
#   1. 对同一批图像，分别提取「完整图」和「中心裁剪图」的 per-layer embeddings
#   2. 对每个 sample 的 (L, d) 矩阵做 DMD fusion → 得到 (d,) 向量
#   3. 计算 full vs cropped 的 cosine 相似度
#   4. 统计并可视化结果

# 用法:
#   python ablation_vision_crop.py \
#       --model_name facebook/dinov2-base \
#       --img_root   data/image_data/images \
#       --events_dir data/image_data/ds004192-download \
#       --save_root  results/ablation_vision \
#       --crop_ratio 0.5 \
#       --device     mps
# """

# import argparse
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import numpy as np
# import matplotlib.pyplot as plt

# from glob import glob
# from tqdm import tqdm
# from PIL import Image

# from utils.dmd import fuse_layers_single_soft_dmd
# from utils.encoder.image_encoder import load_image_model, get_image_embeddings_from_pil


# # ─────────────────────────────────────────────────────────────────────────────
# # 1. 图像加载与裁剪
# # ─────────────────────────────────────────────────────────────────────────────

# def random_patch_keep_pil(img: Image.Image, ratio: float, patch_size: int = 16) -> Image.Image:
#     """
#     随机保留 ratio 比例的 patch，其余 patch 用图像均值填充。
#     和语音/语言的随机散点保留对应。
#     """
#     img = img.copy()
#     W, H = img.size
#     mean_color = (0, 0, 0)

#     patches_x = W // patch_size
#     patches_y = H // patch_size
#     n_patches = patches_x * patches_y
#     n_keep = max(1, int(n_patches * ratio))

#     keep_idx = set(np.random.choice(n_patches, size=n_keep, replace=False))
#     pixels = img.load()
#     for idx in range(n_patches):
#         if idx not in keep_idx:
#             px = (idx % patches_x) * patch_size
#             py = (idx // patches_x) * patch_size
#             for x in range(px, min(px + patch_size, W)):
#                 for y in range(py, min(py + patch_size, H)):
#                     pixels[x, y] = mean_color
#     return img


# def load_images(paths: list) -> list:
#     return [Image.open(p).convert("RGB") for p in paths]


# # ─────────────────────────────────────────────────────────────────────────────
# # 2. Embedding 提取（直接传 PIL list，无临时文件）
# # ─────────────────────────────────────────────────────────────────────────────

# def get_layer_embeddings(extractor, model, images: list, device: str, batch_size: int = 8) -> np.ndarray:
#     """
#     输入: PIL Image list，长度 N
#     输出: (L, N, d)  —— L层, N样本, d特征维
#     """
#     X_layers = get_image_embeddings_from_pil(
#         extractor, model, images,
#         device=device, cls_only=True,
#         batch_size=batch_size,
#     )
#     # X_layers: list of (N, d)，长度 L → stack → (L, N, d)
#     return np.stack(X_layers, axis=0).astype(np.float32)


# # ─────────────────────────────────────────────────────────────────────────────
# # 3. DMD Fusion：对每个 sample 跨层融合
# # ─────────────────────────────────────────────────────────────────────────────

# def dmd_fuse_samples(X_LNd: np.ndarray, k: int = 3, center: float = 1.0, sigma: float = 0.1) -> np.ndarray:
#     """
#     输入: (L, N, d)
#     输出: (N, d)  —— 每个 sample 跨层 soft DMD fusion
#     center: 目标谱半径，0~1之间选瞬态模式，1附近选稳态模式
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
#     """A, B: (N, d) → per-sample cosine sim, shape (N,)"""
#     A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
#     B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
#     return (A_norm * B_norm).sum(axis=1)


# def cosine_per_layer(X_full: np.ndarray, X_crop: np.ndarray) -> np.ndarray:
#     """X_full, X_crop: (L, N, d) → per-layer mean cosine sim, shape (L,)"""
#     return np.array([
#         cosine_sim(X_full[l], X_crop[l]).mean()
#         for l in range(X_full.shape[0])
#     ])


# # ─────────────────────────────────────────────────────────────────────────────
# # 5. 可视化
# # ─────────────────────────────────────────────────────────────────────────────

# def plot_results(layer_sims, dmd_sim_mean, dmd_sim_std, model_name, crop_ratio, save_path):
#     L = len(layer_sims)
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
#     fig.suptitle(
#         f"Context Ablation — {model_name}\n(center crop ratio={crop_ratio})",
#         fontsize=13, fontweight='bold', y=1.01
#     )

#     ax = axes[0]
#     ax.plot(range(L), layer_sims, 'o-', color='#2563eb', lw=2, ms=5)
#     ax.axhline(y=layer_sims.mean(), color='gray', ls='--', lw=1.2,
#                label=f'mean={layer_sims.mean():.3f}')
#     ax.set_xlabel("Layer index", fontsize=11)
#     ax.set_ylabel("Cosine similarity (full vs cropped)", fontsize=11)
#     ax.set_title("Per-layer similarity", fontsize=12)
#     ax.set_ylim(0, 1.05)
#     ax.legend(fontsize=9)
#     ax.grid(True, alpha=0.3)

#     ax2 = axes[1]
#     ax2.bar(['DMD-fused'], [dmd_sim_mean], yerr=[dmd_sim_std],
#             color='#16a34a', alpha=0.8, capsize=8, width=0.4,
#             error_kw={'lw': 2})
#     ax2.set_ylim(0, 1.05)
#     ax2.set_ylabel("Cosine similarity", fontsize=11)
#     ax2.set_title(f"After DMD fusion\nmean={dmd_sim_mean:.3f} ± {dmd_sim_std:.3f}", fontsize=12)
#     ax2.grid(True, alpha=0.3, axis='y')

#     verdict = (
#         "⚠ High sim → instantaneous mode carries little global context"
#         if dmd_sim_mean > 0.85
#         else "✓ Low sim → instantaneous mode retains global context"
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
#     model_name, img_root, save_root,
#     crop_ratio=0.5, dmd_k=3, dmd_sigma=0.1,
#     device="mps", batch_size=8, max_samples=500,
# ):
#     os.makedirs(save_root, exist_ok=True)
#     model_tag = model_name.split("/")[-1]

#     # ── 收集图像路径 ──────────────────────────────────────────────────────────
#     print("[1/5] Collecting image paths ...")
#     img_paths = []
#     for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG"):
#         img_paths.extend(glob(os.path.join(img_root, "**", ext), recursive=True))

#     img_paths = sorted(set(img_paths))[:max_samples]
#     print(f"   → {len(img_paths)} unique images")
#     if not img_paths:
#         raise ValueError(f"No images found in {img_root}")

#     # ── 加载图像 ──────────────────────────────────────────────────────────────
#     print("[2/5] Loading images ...")
#     full_imgs    = load_images(img_paths)
#     cropped_imgs = [random_patch_keep_pil(img, crop_ratio) for img in full_imgs]

#     # ── 提取 embeddings ───────────────────────────────────────────────────────
#     print("[3/5] Extracting embeddings ...")
#     extractor, model = load_image_model(model_name, device=device)

#     print("   full images:")
#     X_full = get_layer_embeddings(extractor, model, full_imgs, device, batch_size)
#     print(f"   shape: {X_full.shape}  (L, N, d)")

#     print("   cropped images:")
#     X_crop = get_layer_embeddings(extractor, model, cropped_imgs, device, batch_size)

#     # ── Per-layer cosine sim ──────────────────────────────────────────────────
#     print("[4/5] Per-layer cosine similarity ...")
#     layer_sims = cosine_per_layer(X_full, X_crop)
#     for l, s in enumerate(layer_sims):
#         print(f"   Layer {l:02d}: {s:.4f}")

#     # ── DMD Fusion：同时跑所有 center 值 ────────────────────────────────────
#     print("[5/5] No-DMD baseline + DMD fusion across all centers ...")

#     # ── 无DMD基线：所有层均值池化 ────────────────────────────────────────────
#     X_full_mean  = X_full.mean(axis=0)        # (N, d)
#     X_other_mean = X_crop.mean(axis=0) # (N, d)
#     sim_nodmd = cosine_sim(X_full_mean, X_other_mean)
#     nodmd_mean = float(sim_nodmd.mean())
#     nodmd_std  = float(sim_nodmd.std())
#     print(f"   no_dmd (mean pool): {nodmd_mean:.4f} ± {nodmd_std:.4f}")

#     centers = [0, 0.2, 0.4, 0.6, 0.8,1.0,1.2,1.4]
#     results_by_center = {}

#     for c in centers:
#         fused_full = dmd_fuse_samples(X_full, k=dmd_k, center=c, sigma=dmd_sigma)
#         fused_crop = dmd_fuse_samples(X_crop, k=dmd_k, center=c, sigma=dmd_sigma)
#         per_sample_sim = cosine_sim(fused_full, fused_crop)
#         sim_mean = float(per_sample_sim.mean())
#         sim_std  = float(per_sample_sim.std())
#         results_by_center[c] = {"mean": sim_mean, "std": sim_std}
#         print(f"   center={c:.1f}: {sim_mean:.4f} ± {sim_std:.4f}")

#     tag = f"{model_tag}_crop{crop_ratio}"
#     npz_path = os.path.join(save_root, f"{tag}.npz")
#     save_dict = {"layer_sims": layer_sims}
#     for c, r in results_by_center.items():
#         save_dict[f"sim_mean_c{c}"] = np.array([r["mean"]])
#         save_dict[f"sim_std_c{c}"]  = np.array([r["std"]])
#     np.savez(npz_path, **save_dict)
#     print(f"   Saved → {npz_path}")

#     fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
#     fig.suptitle(f"Vision Context Ablation — {model_tag}\n(crop_ratio={crop_ratio})",
#                  fontsize=13, fontweight="bold", y=1.01)

#     ax = axes[0]
#     ax.plot(range(len(layer_sims)), layer_sims, "o-", color="#2563eb", lw=2, ms=5)
#     ax.axhline(y=layer_sims.mean(), color="gray", ls="--", lw=1.2,
#                label=f"mean={layer_sims.mean():.3f}")
#     ax.set_xlabel("Layer index"); ax.set_ylabel("Cosine similarity (full vs masked)")
#     ax.set_title("Per-layer similarity"); ax.set_ylim(0, 1.05)
#     ax.legend(); ax.grid(True, alpha=0.3)

#     ax2 = axes[1]
#     c_vals = list(results_by_center.keys())
#     means  = [results_by_center[c]["mean"] for c in c_vals]
#     stds   = [results_by_center[c]["std"]  for c in c_vals]
#     colors = ["#ef4444" if m < 0 else "#3b82f6" for m in means]
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
#     print(f"\nVision Ablation  ratio={crop_ratio}")
#     print(_sep); print(_hdr); print(_sep); print(_row); print(_sep + "\n")

#     return {
#         "model": model_name,
#         "crop_ratio": crop_ratio,
#         "n_samples": len(img_paths),
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
#     parser.add_argument("--model_name",  type=str,   default="facebook/dinov2-base")
#     parser.add_argument("--img_root",    type=str,   default="data/image_data/images")
#     parser.add_argument("--save_root",   type=str,   default="results/ablation_vision")
#     parser.add_argument("--crop_ratio",  type=float, default=0.25)
#     parser.add_argument("--dmd_k",       type=int,   default=3)
#     parser.add_argument("--dmd_sigma",   type=float, default=0.1)
#     parser.add_argument("--device",      type=str,   default="mps")
#     parser.add_argument("--batch_size",  type=int,   default=8)
#     parser.add_argument("--max_samples", type=int,   default=500)
#     parser.add_argument("--all_models",  action="store_true")
#     args = parser.parse_args()

#     ALL_VISION_MODELS = [
#         "facebook/dino-vitb16",
#         "facebook/dinov2-base",
#         "facebook/dinov2-large",
#         "google/vit-base-patch16-224-in21k",
#         "facebook/vit-mae-base",
#         "facebook/vit-msn-base",
#     ]
#     models_to_run = ALL_VISION_MODELS if args.all_models else [args.model_name]

#     all_results = []
#     for mn in models_to_run:
#         print(f"\n{'='*60}\n  Model: {mn}\n{'='*60}")
#         res = run_ablation(
#             model_name=mn,
#             img_root=args.img_root,
#             save_root=args.save_root,
#             crop_ratio=args.crop_ratio,
#             dmd_k=args.dmd_k,
#             dmd_sigma=args.dmd_sigma,
#             device=args.device,
#             batch_size=args.batch_size,
#             max_samples=args.max_samples,
#         )
#         all_results.append(res)

#     print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
#     print(f"{'Model':<35} {'DMD sim':>10}  {'std':>8}  Interpretation")
#     print("-" * 72)
#     for r in all_results:
#         tag = r["model"].split("/")[-1]
#         verdict = "HIGH → low context" if r["dmd_sim_mean"] > 0.85 else "LOW  → rich context"
#         print(f"{tag:<35} {r['dmd_sim_mean']:>10.4f}  {r['dmd_sim_std']:>8.4f}  {verdict}")



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

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from PIL import Image

from core.dmd import fuse_layers_single_soft_dmd
from core.encoder.image_encoder import load_image_model, get_image_embeddings_from_pil


# ─────────────────────────────────────────────────────────────────────────────
# 1. 图像加载与裁剪
# ─────────────────────────────────────────────────────────────────────────────

def random_patch_keep_pil(img: Image.Image, ratio: float, patch_size: int = 16) -> Image.Image:
    """
    随机保留 ratio 比例的 patch，其余 patch 用图像均值填充。
    和语音/语言的随机散点保留对应。
    """
    img = img.copy()
    W, H = img.size
    mean_color = (0, 0, 0)

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
                    pixels[x, y] = mean_color
    return img


def load_images(paths: list) -> list:
    return [Image.open(p).convert("RGB") for p in paths]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Embedding 提取（直接传 PIL list，无临时文件）
# ─────────────────────────────────────────────────────────────────────────────

def get_layer_embeddings(extractor, model, images: list, device: str, batch_size: int = 8) -> np.ndarray:
    """
    输入: PIL Image list，长度 N
    输出: (L, N, d)  —— L层, N样本, d特征维
    """
    X_layers = get_image_embeddings_from_pil(
        extractor, model, images,
        device=device, cls_only=True,
        batch_size=batch_size,
    )
    # X_layers: list of (N, d)，长度 L → stack → (L, N, d)
    return np.stack(X_layers, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DMD Fusion：对每个 sample 跨层融合
# ─────────────────────────────────────────────────────────────────────────────

def dmd_fuse_samples(X_LNd: np.ndarray, k: int = 3, center: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    """
    输入: (L, N, d)
    输出: (N, d)  —— 每个 sample 跨层 soft DMD fusion
    center: 目标谱半径，0~1之间选瞬态模式，1附近选稳态模式
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
    """A, B: (N, d) → per-sample cosine sim, shape (N,)"""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (A_norm * B_norm).sum(axis=1)


def cosine_per_layer(X_full: np.ndarray, X_crop: np.ndarray) -> np.ndarray:
    """X_full, X_crop: (L, N, d) → per-layer mean cosine sim, shape (L,)"""
    return np.array([
        cosine_sim(X_full[l], X_crop[l]).mean()
        for l in range(X_full.shape[0])
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 5. 可视化
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(layer_sims, dmd_sim_mean, dmd_sim_std, model_name, crop_ratio, save_path):
    L = len(layer_sims)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Context Ablation — {model_name}\n(center crop ratio={crop_ratio})",
        fontsize=13, fontweight='bold', y=1.01
    )

    ax = axes[0]
    ax.plot(range(L), layer_sims, 'o-', color='#2563eb', lw=2, ms=5)
    ax.axhline(y=layer_sims.mean(), color='gray', ls='--', lw=1.2,
               label=f'mean={layer_sims.mean():.3f}')
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Cosine similarity (full vs cropped)", fontsize=11)
    ax.set_title("Per-layer similarity", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(['DMD-fused'], [dmd_sim_mean], yerr=[dmd_sim_std],
            color='#16a34a', alpha=0.8, capsize=8, width=0.4,
            error_kw={'lw': 2})
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cosine similarity", fontsize=11)
    ax2.set_title(f"After DMD fusion\nmean={dmd_sim_mean:.3f} ± {dmd_sim_std:.3f}", fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    verdict = (
        "⚠ High sim → instantaneous mode carries little global context"
        if dmd_sim_mean > 0.85
        else "✓ Low sim → instantaneous mode retains global context"
    )
    fig.text(0.5, -0.04, verdict, ha='center', fontsize=11,
             color='#dc2626' if dmd_sim_mean > 0.85 else '#15803d',
             fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] → {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 图像对比可视化（原图 vs 遮蔽图）
# ─────────────────────────────────────────────────────────────────────────────

def save_image_grid(
    full_imgs: list,
    masked_imgs: list,
    save_path: str,
    n_show: int = 8,
    img_size: int = 128,
):
    """
    将原图与遮蔽图并排保存为网格图。
    每行两列：左=原图，右=遮蔽图。
    n_show: 最多展示多少对图像
    img_size: 每张缩略图的边长（像素）
    """
    n = min(n_show, len(full_imgs))
    pad = 4  # 图像之间的间距（像素）
    col_w = img_size * 2 + pad        # 每对图像的宽度（原图 + 遮蔽图）
    row_h = img_size + pad
    n_cols = 2                        # 每行展示几对（可调）
    n_rows = (n + n_cols - 1) // n_cols

    canvas_w = n_cols * (col_w + pad) + pad
    canvas_h = n_rows * row_h + pad + 20  # +20 留给顶部标题

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))

    for i in range(n):
        row = i // n_cols
        col = i % n_cols

        orig  = full_imgs[i].resize((img_size, img_size), Image.LANCZOS)
        masked = masked_imgs[i].resize((img_size, img_size), Image.LANCZOS)

        x_off = pad + col * (col_w + pad)
        y_off = pad + 20 + row * row_h  # +20 为标题留空间

        canvas.paste(orig,   (x_off, y_off))
        canvas.paste(masked, (x_off + img_size + pad, y_off))

    # 用 matplotlib 加标题并保存（避免 PIL 字体依赖问题）
    fig, ax = plt.subplots(figsize=(canvas_w / 72, canvas_h / 72), dpi=72)
    ax.imshow(np.array(canvas))
    ax.set_title("Left: original  |  Right: masked", fontsize=9, pad=4)
    ax.axis("off")
    plt.tight_layout(pad=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Grid] → {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    model_name, img_root, save_root,
    crop_ratio=0.5, dmd_k=3, dmd_sigma=0.1,
    device="mps", batch_size=8, max_samples=500,
):
    os.makedirs(save_root, exist_ok=True)
    model_tag = model_name.split("/")[-1]

    # ── 收集图像路径 ──────────────────────────────────────────────────────────
    print("[1/5] Collecting image paths ...")
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG"):
        img_paths.extend(glob(os.path.join(img_root, "**", ext), recursive=True))

    img_paths = sorted(set(img_paths))[:max_samples]
    print(f"   → {len(img_paths)} unique images")
    if not img_paths:
        raise ValueError(f"No images found in {img_root}")

    # ── 加载图像 ──────────────────────────────────────────────────────────────
    print("[2/5] Loading images ...")
    full_imgs    = load_images(img_paths)
    cropped_imgs = [random_patch_keep_pil(img, crop_ratio) for img in full_imgs]

    # ── 保存图像对比网格 ──────────────────────────────────────────────────────
    print("[2.5/5] Saving image comparison grid ...")
    grid_path = os.path.join(save_root, f"{model_tag}_crop{crop_ratio}_sample_grid.png")
    save_image_grid(
        full_imgs, cropped_imgs,
        save_path=grid_path,
        n_show=4,       # 展示前16对，可按需调整
        img_size=128,
    )

    # ── 提取 embeddings ───────────────────────────────────────────────────────
    print("[3/5] Extracting embeddings ...")
    extractor, model = load_image_model(model_name, device=device)

    print("   full images:")
    X_full = get_layer_embeddings(extractor, model, full_imgs, device, batch_size)
    print(f"   shape: {X_full.shape}  (L, N, d)")

    print("   cropped images:")
    X_crop = get_layer_embeddings(extractor, model, cropped_imgs, device, batch_size)

    # ── Per-layer cosine sim ──────────────────────────────────────────────────
    print("[4/5] Per-layer cosine similarity ...")
    layer_sims = cosine_per_layer(X_full, X_crop)
    for l, s in enumerate(layer_sims):
        print(f"   Layer {l:02d}: {s:.4f}")

    # ── DMD Fusion：同时跑所有 center 值 ────────────────────────────────────
    print("[5/5] No-DMD baseline + DMD fusion across all centers ...")

    # ── 无DMD基线：所有层均值池化 ────────────────────────────────────────────
    X_full_mean  = X_full.mean(axis=0)        # (N, d)
    X_other_mean = X_crop.mean(axis=0) # (N, d)
    sim_nodmd = cosine_sim(X_full_mean, X_other_mean)
    nodmd_mean = float(sim_nodmd.mean())
    nodmd_std  = float(sim_nodmd.std())
    print(f"   no_dmd (mean pool): {nodmd_mean:.4f} ± {nodmd_std:.4f}")

    centers = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    results_by_center = {}

    for c in centers:
        fused_full = dmd_fuse_samples(X_full, k=dmd_k, center=c, sigma=dmd_sigma)
        fused_crop = dmd_fuse_samples(X_crop, k=dmd_k, center=c, sigma=dmd_sigma)
        per_sample_sim = cosine_sim(fused_full, fused_crop)
        sim_mean = float(per_sample_sim.mean())
        sim_std  = float(per_sample_sim.std())
        results_by_center[c] = {"mean": sim_mean, "std": sim_std}
        print(f"   center={c:.1f}: {sim_mean:.4f} ± {sim_std:.4f}")

    tag = f"{model_tag}_crop{crop_ratio}"
    npz_path = os.path.join(save_root, f"{tag}.npz")
    save_dict = {"layer_sims": layer_sims}
    for c, r in results_by_center.items():
        save_dict[f"sim_mean_c{c}"] = np.array([r["mean"]])
        save_dict[f"sim_std_c{c}"]  = np.array([r["std"]])
    np.savez(npz_path, **save_dict)
    print(f"   Saved → {npz_path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"Vision Context Ablation — {model_tag}\n(crop_ratio={crop_ratio})",
                 fontsize=13, fontweight="bold", y=1.01)

    ax = axes[0]
    ax.plot(range(len(layer_sims)), layer_sims, "o-", color="#2563eb", lw=2, ms=5)
    ax.axhline(y=layer_sims.mean(), color="gray", ls="--", lw=1.2,
               label=f"mean={layer_sims.mean():.3f}")
    ax.set_xlabel("Layer index"); ax.set_ylabel("Cosine similarity (full vs masked)")
    ax.set_title("Per-layer similarity"); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    c_vals = list(results_by_center.keys())
    means  = [results_by_center[c]["mean"] for c in c_vals]
    stds   = [results_by_center[c]["std"]  for c in c_vals]
    colors = ["#ef4444" if m < 0 else "#3b82f6" for m in means]
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
    print(f"\nVision Ablation  ratio={crop_ratio}")
    print(_sep); print(_hdr); print(_sep); print(_row); print(_sep + "\n")

    return {
        "model": model_name,
        "crop_ratio": crop_ratio,
        "n_samples": len(img_paths),
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
    parser.add_argument("--model_name",  type=str,   default="facebook/dinov2-base")
    parser.add_argument("--img_root",    type=str,   default="data/image_data/images")
    parser.add_argument("--save_root",   type=str,   default="results/ablation_vision")
    parser.add_argument("--crop_ratio",  type=float, default=0.5)
    parser.add_argument("--dmd_k",       type=int,   default=3)
    parser.add_argument("--dmd_sigma",   type=float, default=0.1)
    parser.add_argument("--device",      type=str,   default="mps")
    parser.add_argument("--batch_size",  type=int,   default=1)
    parser.add_argument("--max_samples", type=int,   default=500)
    parser.add_argument("--all_models",  action="store_true")
    args = parser.parse_args()

    ALL_VISION_MODELS = [
        "facebook/dino-vitb16",
        "facebook/dinov2-base",
        "facebook/dinov2-large",
        "google/vit-base-patch16-224-in21k",
        "facebook/vit-mae-base",
        "facebook/vit-msn-base",
    ]
    models_to_run = ALL_VISION_MODELS if args.all_models else [args.model_name]

    all_results = []
    for mn in models_to_run:
        print(f"\n{'='*60}\n  Model: {mn}\n{'='*60}")
        res = run_ablation(
            model_name=mn,
            img_root=args.img_root,
            save_root=args.save_root,
            crop_ratio=args.crop_ratio,
            dmd_k=args.dmd_k,
            dmd_sigma=args.dmd_sigma,
            device=args.device,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        all_results.append(res)

    centers = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    _cw   = 9
    _cols = ["No_DMD"] + ["c=" + str(c) for c in centers]
    _hdr  = "  " + f"{'Model':<30}" + "".join(f"{h:>{_cw}}" for h in _cols)
    _sep  = "-" * len(_hdr)
    print(f"\nVision Ablation Summary")
    print(_sep); print(_hdr); print(_sep)
    for r in all_results:
        tag = r["model"].split("/")[-1]
        _vals = [r["nodmd_mean"]] + [r["results_by_center"][str(c)] for c in centers]
        _row  = "  " + f"{tag:<30}" + "".join(f"{v:>{_cw}.4f}" for v in _vals)
        print(_row)
    print(_sep)