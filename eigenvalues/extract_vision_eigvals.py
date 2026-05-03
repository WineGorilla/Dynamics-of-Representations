"""
视觉 Embedding DMD 特征值提取
==============================
从 embeddings/vision/{model}.npy 提取 DMD 特征值
输出到 neweigvals/vision/{model}.npy

用法:
  CUDA_VISIBLE_DEVICES=0 python eigenvalues/extract_vision_eigvals.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import numpy as np
from tqdm import tqdm
from core.dmd import compute_dmd_eigenvalues


VISION_MODELS = [
    # ── DINOv2 (3) ──
    "dinov2-small",
    "dinov2-base",
    "dinov2-large",

    # ── DINO (2) ──
    "dino-vitb16",
    "dino-vits16",

    # ── BEiT (2) ──
    "beit-base-patch16-224-pt22k-ft22k",
    "beit-large-patch16-224-pt22k-ft22k",

    # ── DeiT (2) ──
    "deit-base-patch16-224",
    "deit-small-patch16-224",

    # ── ViT (2) ──
    "vit-base-patch16-224-in21k",
    "vit-large-patch16-224-in21k",

    # ── ViT-MAE (2) ──
    "vit-mae-base",
    "vit-mae-large",

    # ── ViT-MSN (2) ──
    "vit-msn-base",
    "vit-msn-large",

    # ── Data2Vec-Vision (2) ──
    "data2vec-vision-base",
    "data2vec-vision-large",

    # ── CLIP (3) ──
    "clip-vit-base-patch32",
    "clip-vit-base-patch16",
    "clip-vit-large-patch14",

    # ── Swin (3) ──
    "swin-tiny-patch4-window7-224",
    "swin-small-patch4-window7-224",
    "swin-large-patch4-window7-224",

    # ── SAM (3) ──
    "sam-vit-base",
    "sam-vit-large",
    "sam-vit-huge",

    # ── CNN (10) ──
    "resnet50",
    "resnet101",
    "densenet121",
    "densenet201",
    "efficientnet_b0",
    "efficientnet_b4",
    "convnext_tiny",
    "convnext_base",
    "vgg16",
    "vgg19",
]


def collect_one_vision_model(model, root="embeddings/vision"):
    """
    新结构：每个模型是一个 .npy 文件 (n_layers, n_images, feat_dim)
    对每个 image (时间点) 计算 DMD 特征值
    """
    npy_path = os.path.join(root, f"{model}.npy")

    if not os.path.exists(npy_path):
        print(f"  ⚠ 文件不存在: {npy_path}")
        return np.array([])

    X = np.load(npy_path)  # (L, N, d)
    print(f"[Model] {model}")
    print(f"  shape: {X.shape}")

    if X.ndim != 3:
        print(f"  ⚠ 维度不对: {X.ndim}")
        return np.array([])

    L, N, d = X.shape
    all_eigs = []

    for t in tqdm(range(N), desc=f"  {model}", ncols=80, leave=False):
        eigvals = compute_dmd_eigenvalues(X[:, t, :])
        if eigvals is not None:
            all_eigs.extend(eigvals)

    return np.array(all_eigs)


def main():
    save_root = "neweigvals/vision"
    os.makedirs(save_root, exist_ok=True)

    for model in VISION_MODELS:
        print(f"\nCollecting: {model}")
        try:
            eigvals = collect_one_vision_model(model=model, root="embeddings/vision")
            save_path = os.path.join(save_root, f"{model}.npy")
            np.save(save_path, eigvals)
            print(f"  ✅ saved → {save_path}  shape={eigvals.shape}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()