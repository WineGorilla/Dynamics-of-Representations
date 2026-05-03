"""
语言 Embedding DMD 特征值提取
==============================
从 embeddings/language/{model_tag}/*.npy 提取 DMD 特征值
输出到 neweigvals/language/{model_tag}.npy

用法:
  python eigenvalues/extract_language_eigvals.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from core.dmd import compute_dmd_eigenvalues


LANGUAGE_MODELS = [
    # ── MiniLM / MPNet (4) ──
    "MiniLM-L6-H384-uncased",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "mpnet-base",

    # ── ALBERT (3) ──
    "albert-base-v2",
    "albert-large-v2",
    "albert-xlarge-v2",

    # ── BERT (5) ──
    "bert-base-cased",
    "bert-base-multilingual-cased",
    "bert-base-uncased",
    "bert-large-cased",
    "bert-large-uncased",

    # ── CamemBERT (1) ──
    "camembert-base",

    # ── ConvBERT (2) ──
    "conv-bert-base",
    "conv-bert-medium-small",

    # ── Data2Vec-Text (1) ──
    "data2vec-text-base",

    # ── DeBERTa (2) ──
    "deberta-base",
    "deberta-large",

    # ── DistilBERT (2) ──
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased",

    # ── DistilRoBERTa (1) ──
    "distilroberta-base",

    # ── ELECTRA (3) ──
    "electra-base-discriminator",
    "electra-large-discriminator",
    "electra-small-discriminator",

    # ── ERNIE (2) ──
    "ernie-2.0-base-en",
    "ernie-2.0-large-en",

    # ── iBERT (1) ──
    "ibert-roberta-base",

    # ── RemBERT (1) ──
    "rembert",

    # ── RoBERTa (2) ──
    "roberta-base",
    "roberta-large",

    # ── SqueezeBERT (1) ──
    "squeezebert-uncased",

    # ── T5 (1) ──
    "t5-small",

    # ── XLM-RoBERTa (2) ──
    "xlm-roberta-base",
    "xlm-roberta-large",

    # ── XLNet (2) ──
    "xlnet-base-cased",
    "xlnet-large-cased",
]


def collect_one_language_model(model, root="embeddings/language"):
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
    save_root = "neweigvals/language"
    os.makedirs(save_root, exist_ok=True)

    for model in LANGUAGE_MODELS:
        print(f"\nCollecting: {model}")
        try:
            eigvals = collect_one_language_model(model=model, root="embeddings/language")
            save_path = os.path.join(save_root, f"{model}.npy")
            np.save(save_path, eigvals)
            print(f"  ✅ saved → {save_path}  shape={eigvals.shape}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()