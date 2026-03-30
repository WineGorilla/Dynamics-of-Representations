import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def plot_eigval_distribution(eigval_dir, modality="vision", save_path=None):
    """
    读取指定目录下所有模型的特征值 .npy 文件，
    画出每个模型的 |λ| 分布（histogram）。

    eigval_dir: e.g. "processed/eigvals/vision"
    modality:   用于标题显示
    """
    npy_files = sorted(glob(os.path.join(eigval_dir, "*.npy")))
    if not npy_files:
        print(f"No .npy files found in {eigval_dir}")
        return

    n_models = len(npy_files)
    ncols = 4
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for idx, fpath in enumerate(npy_files):
        model_name = os.path.basename(fpath).replace(".npy", "").replace("_", "/", 1)
        eigvals = np.load(fpath, allow_pickle=True)

        rho = np.abs(eigvals)  # 特征值的模

        ax = axes[idx]
        ax.hist(rho, bins=50, range=(0, 2), color="#3b82f6", alpha=0.8, edgecolor="none")
        ax.axvline(x=1.0, color="red", linestyle="--", lw=1.5, label="λ=1")
        ax.set_title(model_name.split("/")[-1], fontsize=9)
        ax.set_xlabel("|λ|", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        mean_rho = rho.mean()
        frac_near1 = ((rho > 0.9) & (rho < 1.1)).mean()
        ax.set_title(
            f"{model_name.split('/')[-1]}\n"
            f"mean={mean_rho:.2f}  near1={frac_near1*100:.1f}%",
            fontsize=8
        )

    # 隐藏多余的子图
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{modality} models — |λ| distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = f"eigval_dist_{modality}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    # 视觉
    plot_eigval_distribution(
        eigval_dir="processed/eigvals/vision",
        modality="vision",
        save_path="eigval_dist_vision.png"
    )

    # # 语言
    # plot_eigval_distribution(
    #     eigval_dir="processed/eigvals/language",
    #     modality="language",
    #     save_path="eigval_dist_language.png"
    # )

    # plot_eigval_distribution(
    #     eigval_dir="processed/eigvals/audio",
    #     modality="audio",
    #     save_path="eigval_dist_audio.png"
    # )