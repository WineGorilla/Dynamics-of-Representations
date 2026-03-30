import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import os
from glob import glob
from tqdm import tqdm

from core.dmd import fuse_layers_single_time_dmd



def fuse_all_audio_models(
    root="data/audio",
    k=3
):
    design_root = os.path.join(root, "design_matrix")
    fused_root  = os.path.join(root, "design_matrix_dmd_mean")

    os.makedirs(fused_root, exist_ok=True)

    # === 遍历每一个模型文件夹 ===
    model_names = sorted(
        d for d in os.listdir(design_root)
        if os.path.isdir(os.path.join(design_root, d))
    )

    print(f"Found {len(model_names)} audio models")

    for model in model_names:
        print(f"\n[Model] {model}")

        in_dir  = os.path.join(design_root, model)
        out_dir = os.path.join(fused_root, model)
        os.makedirs(out_dir, exist_ok=True)

        npy_files = sorted(glob(os.path.join(in_dir, "*.npy")))
        print(f"  Files: {len(npy_files)}")

        for in_path in tqdm(npy_files):
            fname = os.path.basename(in_path)
            out_path = os.path.join(out_dir, fname)

            X = np.load(in_path)      # (L, T, d)
            if X.ndim != 3:
                print(f"Skip {fname}, shape={X.shape}")
                continue

            L, T, d = X.shape
            fused = np.zeros((T, d), dtype=np.float32)

            for t in range(T):
                fused[t] = fuse_layers_single_time_dmd(X[:, t, :],r=k)

            np.save(out_path, fused)

fuse_all_audio_models(
    root="data/audio",
    k=1
)




