import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
from glob import glob
from tqdm import tqdm

from core.dmd import fuse_layers_single_time_dmd

def fuse_all_img_models(
    root="data/img",
    k=3
):
    design_root = os.path.join(root, "design_matrix")
    fused_root  = os.path.join(root, "design_matrix_dmd_mean")
    os.makedirs(fused_root, exist_ok=True)

    # === 每一个模型 ===
    model_names = sorted(
        d for d in os.listdir(design_root)
        if os.path.isdir(os.path.join(design_root, d))
    )

    print(f"Found {len(model_names)} img models")

    for model in model_names:
        print(f"\n[Model] {model}")

        in_model_dir  = os.path.join(design_root, model)
        out_model_dir = os.path.join(fused_root, model)

        npy_files = glob(
            os.path.join(in_model_dir, "sub-*", "ses-*", "*.npy")
        )

        print(f"  Files: {len(npy_files)}")

        for in_path in tqdm(npy_files):
            rel_path = os.path.relpath(in_path, in_model_dir)
            out_path = os.path.join(out_model_dir, rel_path)

            X = np.load(in_path)      # (L, T, d)
            if X.ndim != 3:
                print(f"Skip {in_path}, shape={X.shape}")
                continue

            L, T, d = X.shape
            fused = np.zeros((T, d), dtype=np.float32)

            for t in range(T):
                fused[t] = fuse_layers_single_time_dmd(X[:, t, :],r=k)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, fused)

fuse_all_img_models(
    root="data/img",
    k=1
)
