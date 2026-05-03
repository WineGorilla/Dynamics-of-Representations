"""
音频 Embedding DMD 特征值提取
==============================
从 embeddings/audio/{model}/*.npy 提取 DMD 特征值
输出到 processed_new/eigvals/audio/{model}.npy

用法:
  CUDA_VISIBLE_DEVICES=1 python eigenvalues/extract_audio_eigvals.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
from glob import glob
from tqdm import tqdm
from core.dmd import compute_dmd_eigenvalues


AUDIO_MODELS = [
    # ── Data2Vec-Audio (4) ──
    "data2vec-audio-base",
    "data2vec-audio-base-960h",
    "data2vec-audio-large",
    "data2vec-audio-large-960h",

    # ── HuBERT (4) ──
    "hubert-base-ls960",
    "hubert-base-superb-ks",
    "hubert-large-ls960-ft",
    "hubert-xlarge-ls960-ft",

    # ── SEW / SEW-D (6) ──
    "sew-d-mid-100k",
    "sew-d-small-100k",
    "sew-d-tiny-100k",
    "sew-mid-100k",
    "sew-small-100k",
    "sew-tiny-100k",

    # ── UniSpeech / UniSpeech-SAT (4) ──
    "unispeech-large-1500h-cv",
    "unispeech-sat-base",
    "unispeech-sat-base-plus",
    "unispeech-sat-large",

    # ── Wav2Vec2-BERT (1) ──
    "w2v-bert-2.0",

    # ── Wav2Vec2 (10) ──
    "wav2vec2-base",
    "wav2vec2-base-960h",
    "wav2vec2-base-superb-ks",
    "wav2vec2-conformer-rel-pos-large",
    "wav2vec2-conformer-rope-large-960h-ft",
    "wav2vec2-large",
    "wav2vec2-large-960h",
    "wav2vec2-large-xlsr-53",
    "wav2vec2-xls-r-1b",
    "wav2vec2-xls-r-300m",

    # ── WavLM (3) ──
    "wavlm-base",
    "wavlm-base-plus",
    "wavlm-large",

    # ── Whisper (4) ──
    "whisper-base",
    "whisper-medium",
    "whisper-small",
    "whisper-tiny",
]


def collect_one_audio_model(model, root="embeddings/audio"):
    in_model_dir = os.path.join(root, model)
    npy_files = sorted(glob(os.path.join(in_model_dir, "*.npy")))

    print(f"[Model] {model}")
    print(f"  Files: {len(npy_files)}")

    all_eigs = []
    for in_path in tqdm(npy_files, desc=f"  {model}", ncols=80, leave=False):
        X = np.load(in_path)  # (L, T, d)
        if X.ndim != 3:
            continue
        L, T, d = X.shape
        for t in range(T):
            eigvals = compute_dmd_eigenvalues(X[:, t, :])
            if eigvals is not None:
                all_eigs.extend(eigvals)

    return np.array(all_eigs)


def main():
    save_root = "neweigvals/audio"
    os.makedirs(save_root, exist_ok=True)

    for model in AUDIO_MODELS:
        print(f"\nCollecting: {model}")
        try:
            eigvals = collect_one_audio_model(model=model, root="embeddings/audio")
            save_path = os.path.join(save_root, f"{model}.npy")
            np.save(save_path, eigvals)
            print(f"  ✅ saved → {save_path}  shape={eigvals.shape}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()