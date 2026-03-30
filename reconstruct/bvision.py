import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
from core.dmd import fuse_layers_single_soft_dmd

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def extract_hidden_states_no_mask(model, pixel_values):
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, output_hidden_states=True)
    return outputs.hidden_states  # tuple of (1, n_patches+1, D)


def compare_centers(image_path, centers=[0.0, 0.5, 1.0], device="mps",
                    model_name="facebook/dinov2-base"):
    print(f"Loading {model_name} ...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    img_orig     = Image.open(image_path).convert("RGB")
    inputs       = processor(images=img_orig, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]

    hidden_states = extract_hidden_states_no_mask(model, pixel_values)

    # 提取 patch token（去掉 CLS）
    def get_patch(h):
        x = h.cpu().numpy()
        return x[0, 1:, :] if x.ndim == 3 else x[1:, :]

    patch_tokens = np.stack([get_patch(h) for h in hidden_states], axis=0)  # (L, N, D)
    n_patches    = patch_tokens.shape[1]
    D            = patch_tokens.shape[2]
    grid_size    = int(n_patches ** 0.5)
    patch_size   = 224 // grid_size

    print(f"  grid={grid_size}x{grid_size}, n_patches={n_patches}, D={D}, L={patch_tokens.shape[0]}")

    # DMD fusion 每个 center
    all_fused = {}
    for c in centers:
        fused = np.zeros((n_patches, D), dtype=np.float32)
        for p in range(n_patches):
            fused[p] = fuse_layers_single_soft_dmd(patch_tokens[:, p, :], center=c)
        all_fused[c] = fused
        print(f"  center={c} done")

    # 所有 center 的 token 一起拟合 PCA，保证颜色空间一致
    all_tokens = np.concatenate(list(all_fused.values()), axis=0)
    pca = PCA(n_components=3)
    pca.fit(all_tokens)

    # 最后一层原始 patch token（无 DMD 处理）
    last_layer_tokens = patch_tokens[-1]  # (N, D)

    # 所有 token 一起拟合 PCA（包含原始最后一层，保证颜色空间一致）
    all_tokens_with_raw = np.concatenate(
        [last_layer_tokens] + list(all_fused.values()), axis=0
    )
    pca = PCA(n_components=3)
    pca.fit(all_tokens_with_raw)

    def to_vis(tokens):
        rgb = pca.transform(tokens)
        rgb = (rgb - rgb.min(axis=0)) / (rgb.max(axis=0) - rgb.min(axis=0) + 1e-8)
        return np.kron(rgb.reshape(grid_size, grid_size, 3),
                       np.ones((patch_size, patch_size, 1)))

    # 可视化
    n   = len(centers) + 2  # +1 original image, +1 raw last layer
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(img_orig.resize((224, 224)))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(to_vis(last_layer_tokens))
    axes[1].set_title("No DMD\n(last layer)")
    axes[1].axis("off")

    for idx, c in enumerate(centers):
        axes[idx + 2].imshow(to_vis(all_fused[c]))
        axes[idx + 2].set_title(f"center={c}")
        axes[idx + 2].axis("off")

    plt.suptitle(f"DMD patch token PCA — {model_name.split('/')[-1]}", fontsize=12)
    plt.tight_layout()
    save_path = "dmd_pca_vis1.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    compare_centers(
        image_path="data/image_data/images/bee/bee_02s.jpg",
        centers=[0.0, 0.5, 1.0],
        device="mps",
        model_name="facebook/dinov2-base"
    )