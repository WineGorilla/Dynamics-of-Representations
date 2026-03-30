import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from core.dmd import fuse_layers_single_soft_dmd


def compare_centers_lang(text, centers=[0.0, 0.5, 1.0], device="mps",
                          model_name="bert-base-uncased"):
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    # tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    tokens_str = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple (L+1,) each (1, T, D)

    # (L, T, D)
    token_matrix = np.stack(
        [h[0].cpu().numpy() for h in hidden_states], axis=0
    )

    T = token_matrix.shape[1]
    D = token_matrix.shape[2]
    L = token_matrix.shape[0]
    print(f"  tokens={T}, D={D}, L={L}")
    print(f"  words: {tokens_str}")

    # DMD fusion 每个 token
    all_fused = {}
    for c in centers:
        fused = np.zeros((T, D), dtype=np.float32)
        for t in range(T):
            fused[t] = fuse_layers_single_soft_dmd(token_matrix[:, t, :], center=c)
        all_fused[c] = fused
        print(f"  center={c} done")

    # 最后一层原始 token
    last_layer = token_matrix[-1]  # (T, D)

    # 所有 token 一起拟合 PCA
    all_tokens = np.concatenate([last_layer] + list(all_fused.values()), axis=0)
    pca = PCA(n_components=3)
    pca.fit(all_tokens)

    def to_rgb(tokens):
        rgb = pca.transform(tokens)  # (T, 3)
        rgb = (rgb - rgb.min(axis=0)) / (rgb.max(axis=0) - rgb.min(axis=0) + 1e-8)
        return rgb

    # 可视化：每个词一个彩色方块 + 词标签
    n_cols = len(centers) + 1  # original + no_dmd + centers
    fig, axes = plt.subplots(n_cols, 1, figsize=(max(12, T * 0.6), n_cols * 1.5))

    def draw_tokens(ax, rgb, title):
        for i, (word, color) in enumerate(zip(tokens_str, rgb)):
            rect = mpatches.FancyBboxPatch(
                (i, 0), 0.9, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="white", linewidth=1.5
            )
            ax.add_patch(rect)
            # 根据背景亮度选字体颜色
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "black" if brightness > 0.5 else "white"
            ax.text(i + 0.45, 0.4, word, ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")
        ax.set_xlim(0, T)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.axis("off")

    draw_tokens(axes[0], to_rgb(last_layer), "No DMD (last layer)")
    for idx, c in enumerate(centers):
        draw_tokens(axes[idx + 1], to_rgb(all_fused[c]), f"center={c}")

    plt.suptitle(f"DMD token PCA — {model_name.split('/')[-1]}\n\"{text[:80]}\"",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_path = "dmd_lang_pca_vis.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    compare_centers_lang(
        text="The little prince looked at the apple tree for a long time.",
        centers=[0.0, 0.5, 1.0],
        device="mps",
        model_name="bert-base-uncased"
    )