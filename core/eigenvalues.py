from core.dmd import compute_dmd_eigenvalues

import os
import numpy as np
from glob import glob
from tqdm import tqdm


def collect_one_language_model(model="gpt2", root="data/lang"):

    design_root = os.path.join(root, "design_matrix")
    in_model_dir = os.path.join(design_root, model)

    npy_files = sorted(glob(os.path.join(in_model_dir, "*.npy")))

    print(f"[Model] {model}")
    print(f"Files: {len(npy_files)}")

    all_eigs = []

    for in_path in tqdm(npy_files):

        X = np.load(in_path)  # (L, T, d)

        if X.ndim != 3:
            continue

        L, T, d = X.shape

        for t in range(T):

            eigvals = compute_dmd_eigenvalues(X[:, t, :])

            if eigvals is not None:
                all_eigs.extend(eigvals)

    return np.array(all_eigs)


def collect_one_audio_model(model="ast", root="data/audio"):

    design_root = os.path.join(root, "design_matrix")
    in_model_dir = os.path.join(design_root, model)

    npy_files = sorted(glob(os.path.join(in_model_dir, "*.npy")))

    print(f"[Model] {model}")
    print(f"Files: {len(npy_files)}")

    all_eigs = []

    for in_path in tqdm(npy_files):

        X = np.load(in_path)  # (L, T, d)

        if X.ndim != 3:
            continue

        L, T, d = X.shape

        for t in range(T):

            eigvals = compute_dmd_eigenvalues(X[:, t, :])

            if eigvals is not None:
                all_eigs.extend(eigvals)

    return np.array(all_eigs)

def collect_one_img_model(model="clip", root="data/img"):

    design_root = os.path.join(root, "design_matrix")

    in_model_dir  = os.path.join(design_root, model)

    npy_files = glob(
        os.path.join(in_model_dir, "sub-*", "ses-*", "*.npy")
    )

    print(f"[Model] {model}")
    print(f"Files: {len(npy_files)}")

    all_eigs = []

    for in_path in tqdm(npy_files):

        X = np.load(in_path)

        if X.ndim != 3:
            continue

        L, T, d = X.shape

        for t in range(T):

            eigvals = compute_dmd_eigenvalues(X[:, t, :])

            if eigvals is not None:
                all_eigs.extend(eigvals)

    return np.array(all_eigs)