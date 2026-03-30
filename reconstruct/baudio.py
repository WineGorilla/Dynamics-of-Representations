import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa
from transformers import AutoProcessor, AutoModel
from sklearn.decomposition import PCA
from core.dmd import fuse_layers_single_soft_dmd


def compare_centers_audio(audio_path, centers=[0.0, 0.5, 1.0], device="cpu",
                           model_name="facebook/wav2vec2-base-960h",
                           sr=16000, max_frames=60):
    print(f"Loading {model_name} ...")
    processor = AutoProcessor.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    # 加载音频，只取前 5 秒避免 OOM
    print(f"Loading audio: {audio_path}")
    y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=5.0)
    print(f"  audio length: {len(y)/sr:.2f}s ({len(y)} samples)")

    inputs = processor(y, sampling_rate=sr, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple (L+1,) each (1, T, D)

    # (L, T, D)
    token_matrix = np.stack(
        [h[0].cpu().numpy() for h in hidden_states], axis=0
    )

    L, T, D = token_matrix.shape
    print(f"  L={L}, T={T}, D={D}")

    # 截取前 max_frames 帧，避免太长
    T = min(T, max_frames)
    token_matrix = token_matrix[:, :T, :]

    # 时间轴标签（秒）
    frame_duration = 0.02  # wav2vec2 每帧约 20ms
    time_labels = [f"{i * frame_duration:.2f}s" for i in range(T)]

    # DMD fusion 每个时间帧
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
        rgb = pca.transform(tokens)
        rgb = (rgb - rgb.min(axis=0)) / (rgb.max(axis=0) - rgb.min(axis=0) + 1e-8)
        return rgb

    # 可视化
    n_rows = len(centers) + 1  # no_dmd + centers
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(14, T * 0.25), n_rows * 1.8))

    def draw_frames(ax, rgb, title):
        for i, color in enumerate(rgb):
            rect = mpatches.FancyBboxPatch(
                (i, 0), 0.9, 0.8,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor="white", linewidth=1
            )
            ax.add_patch(rect)

        # 每隔 10 帧标一个时间
        for i in range(0, T, 10):
            ax.text(i + 0.45, -0.15, time_labels[i],
                    ha="center", va="top", fontsize=7, color="gray")

        ax.set_xlim(0, T)
        ax.set_ylim(-0.3, 1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.axis("off")

    draw_frames(axes[0], to_rgb(last_layer), "No DMD (last layer)")
    for idx, c in enumerate(centers):
        draw_frames(axes[idx + 1], to_rgb(all_fused[c]), f"center={c}")

    audio_name = os.path.basename(audio_path)
    plt.suptitle(
        f"DMD audio frame PCA — {model_name.split('/')[-1]}\n{audio_name}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    save_path = "dmd_audio_pca_vis.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    import os
    compare_centers_audio(
        audio_path="data/audio_data/stimuli/birthofanation.wav",
        centers=[0.0, 0.5, 1.0],
        device="cpu",
        model_name="facebook/wav2vec2-base-960h",
        max_frames=60
    )