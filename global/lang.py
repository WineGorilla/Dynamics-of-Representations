# """
# Language Context Truncation Ablation
# Full sentence vs randomly kept words (scatter) → CLS embedding cosine similarity
# """

# import argparse
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch

# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModel

# from utils.dmd import fuse_layers_single_soft_dmd


# def load_lang_model(model_name, device):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
#     model.eval()
#     return tokenizer, model


# def random_word_keep(words, keep_ratio, mask_token="[MASK]"):
#     """随机保留 keep_ratio 比例的词，其余替换为 mask_token，保持句子长度不变。"""
#     keep = max(1, int(len(words) * keep_ratio))
#     keep_idx = set(np.random.choice(len(words), size=keep, replace=False))
#     return [words[i] if i in keep_idx else mask_token for i in range(len(words))]


# def build_sentence_windows(words, win_size=50, step=25):
#     sentences = []
#     for i in range(0, max(1, len(words) - win_size + 1), step):
#         sentences.append(words[i:i + win_size])
#     if not sentences:
#         sentences = [words]
#     return sentences


# def get_cls_embedding(sentence, tokenizer, model, device):
#     """单句 → 每层 CLS token embedding, shape (L, d)"""
#     inputs = tokenizer(
#         sentence, return_tensors="pt",
#         truncation=True, max_length=128
#     ).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         hidden_states = outputs.hidden_states
#     return np.stack(
#         [h[:, 0, :].squeeze(0).cpu().numpy() for h in hidden_states],
#         axis=0
#     ).astype(np.float32)


# def get_embeddings_full_vs_masked(words, tokenizer, model, device, keep_ratio=0.5):
#     """
#     完整句子 vs 随机保留 keep_ratio 比例词后的句子。
#     返回 X_full, X_mask: (L, N, d)
#     """
#     sentences = build_sentence_windows(words)
#     full_layers = []
#     masked_layers = []

#     for i, sent_words in enumerate(tqdm(sentences, desc="Encoding")):
#         emb_full = get_cls_embedding(" ".join(sent_words), tokenizer, model, device)
#         full_layers.append(emb_full)

#         cropped_words = random_word_keep(sent_words, keep_ratio)
#         emb_mask = get_cls_embedding(" ".join(cropped_words), tokenizer, model, device)
#         masked_layers.append(emb_mask)

#         if i == 0:
#             print(f"   [debug] full:    {' '.join(sent_words[:8])}...")
#             print(f"   [debug] cropped: {' '.join(cropped_words[:8])}...")

#     # list of (L, d) → stack axis=1 → (L, N, d)
#     X_full = np.stack(full_layers, axis=1).astype(np.float32)
#     X_mask = np.stack(masked_layers, axis=1).astype(np.float32)
#     return X_full, X_mask


# def cosine_per_layer(X_full, X_mask):
#     """(L, N, d) × 2 → per-layer mean cosine sim, shape (L,)"""
#     L, N, d = X_full.shape
#     sims = []
#     for l in range(L):
#         A = X_full[l] / (np.linalg.norm(X_full[l], axis=1, keepdims=True) + 1e-8)
#         B = X_mask[l] / (np.linalg.norm(X_mask[l], axis=1, keepdims=True) + 1e-8)
#         sims.append((A * B).sum(axis=1).mean())
#     return np.array(sims)


# def dmd_fuse_samples(X_LNd, k=3, center=1.0, sigma=0.1):
#     """(L, N, d) → (N, d) via soft DMD fusion"""
#     L, N, d = X_LNd.shape
#     fused = np.zeros((N, d), dtype=np.float32)
#     for n in range(N):
#         fused[n] = fuse_layers_single_soft_dmd(X_LNd[:, n, :], r=k, center=center, sigma=sigma)
#     return fused


# def cosine_sim(A, B):
#     """(N, d) × 2 → (N,)"""
#     A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
#     B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
#     return (A_norm * B_norm).sum(axis=1)


# def plot_results(layer_sims, dmd_sim_mean, dmd_sim_std, model_name, keep_ratio, dmd_center, save_path):
#     L = len(layer_sims)
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
#     fig.suptitle(
#         f"Language Context Ablation — {model_name}\n"
#         f"(keep_ratio={keep_ratio}, dmd_center={dmd_center})",
#         fontsize=13, fontweight='bold', y=1.01
#     )
#     ax = axes[0]
#     ax.plot(range(L), layer_sims, 'o-', color='#ea580c', lw=2, ms=5)
#     ax.axhline(y=layer_sims.mean(), color='gray', ls='--', lw=1.2,
#                label=f'mean={layer_sims.mean():.3f}')
#     ax.set_xlabel("Layer index")
#     ax.set_ylabel("Cosine similarity (full vs cropped)")
#     ax.set_title("Per-layer similarity")
#     ax.set_ylim(0, 1.05)
#     ax.legend()
#     ax.grid(True, alpha=0.3)

#     ax2 = axes[1]
#     ax2.bar(['DMD-fused'], [dmd_sim_mean], yerr=[dmd_sim_std],
#             color='#0284c7', alpha=0.85, capsize=8, width=0.4,
#             error_kw={'lw': 2})
#     ax2.set_ylim(0, 1.05)
#     ax2.set_ylabel("Cosine similarity")
#     ax2.set_title(
#         f"After DMD fusion (center={dmd_center})\n"
#         f"mean={dmd_sim_mean:.3f} ± {dmd_sim_std:.3f}"
#     )
#     ax2.grid(True, alpha=0.3, axis='y')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"[Plot] → {save_path}")


# def run_ablation(model_name, csv_path, save_root, keep_ratio=0.5, device="mps"):
#     os.makedirs(save_root, exist_ok=True)
#     model_tag = model_name.split("/")[-1]

#     print("[1/5] Loading words ...")
#     df = pd.read_csv(csv_path).sort_values(["section", "onset"])
#     sec = sorted(df["section"].unique())[0]
#     words = df[df["section"] == sec]["word"].dropna().astype(str).tolist()
#     words = [w for w in words if w.strip() and w != "nan"]
#     print(f"   → {len(words)} words")

#     print("[2/5] Loading model ...")
#     tokenizer, model = load_lang_model(model_name, device)

#     print("[3/5] Extracting embeddings ...")
#     X_full, X_mask = get_embeddings_full_vs_masked(words, tokenizer, model, device, keep_ratio)
#     print(f"   shape: {X_full.shape}  (L, N, d)")

#     print("[4/5] Per-layer cosine similarity ...")
#     layer_sims = cosine_per_layer(X_full, X_mask)
#     for l, s in enumerate(layer_sims):
#         print(f"   Layer {l:02d}: {s:.4f}")

#     print("[5/5] No-DMD baseline + DMD fusion across all centers ...")

#     # ── 无DMD基线：所有层均值池化 ────────────────────────────────────────────
#     X_full_mean  = X_full.mean(axis=0)        # (N, d)
#     X_other_mean = X_mask.mean(axis=0) # (N, d)
#     sim_nodmd = cosine_sim(X_full_mean, X_other_mean)
#     nodmd_mean = float(sim_nodmd.mean())
#     nodmd_std  = float(sim_nodmd.std())
#     print(f"   no_dmd (mean pool): {nodmd_mean:.4f} ± {nodmd_std:.4f}")

#     centers = [0, 0.2, 0.4, 0.6, 0.8,1.0,1.2,1.4]
#     results_by_center = {}

#     for c in centers:
#         fused_full = dmd_fuse_samples(X_full, center=c)
#         fused_mask = dmd_fuse_samples(X_mask, center=c)
#         per_sample_sim = cosine_sim(fused_full, fused_mask)
#         sim_mean = float(per_sample_sim.mean())
#         sim_std  = float(per_sample_sim.std())
#         results_by_center[c] = {"mean": sim_mean, "std": sim_std}
#         print(f"   center={c:.1f}: {sim_mean:.4f} ± {sim_std:.4f}")

#     tag = f"{model_tag}_keep{keep_ratio}"
#     npz_path = os.path.join(save_root, f"{tag}.npz")
#     save_dict = {"layer_sims": layer_sims}
#     for c, r in results_by_center.items():
#         save_dict[f"sim_mean_c{c}"] = np.array([r["mean"]])
#         save_dict[f"sim_std_c{c}"]  = np.array([r["std"]])
#     np.savez(npz_path, **save_dict)
#     print(f"   Saved → {npz_path}")

#     fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
#     fig.suptitle(f"Language Context Ablation — {model_tag}\n(keep_ratio={keep_ratio})",
#                  fontsize=13, fontweight="bold", y=1.01)

#     ax = axes[0]
#     ax.plot(range(len(layer_sims)), layer_sims, "o-", color="#ea580c", lw=2, ms=5)
#     ax.axhline(y=layer_sims.mean(), color="gray", ls="--", lw=1.2,
#                label=f"mean={layer_sims.mean():.3f}")
#     ax.set_xlabel("Layer index"); ax.set_ylabel("Cosine similarity (full vs masked)")
#     ax.set_title("Per-layer similarity"); ax.set_ylim(0, 1.05)
#     ax.legend(); ax.grid(True, alpha=0.3)

#     ax2 = axes[1]
#     c_vals = list(results_by_center.keys())
#     means  = [results_by_center[c]["mean"] for c in c_vals]
#     stds   = [results_by_center[c]["std"]  for c in c_vals]
#     colors = ["#ef4444" if m < 0 else "#3b82f6" for m in means]
#     ax2.bar([str(c) for c in c_vals], means, yerr=stds, color=colors,
#             alpha=0.8, capsize=6, error_kw={"lw": 1.5})
#     ax2.axhline(y=0, color="black", lw=0.8)
#     ax2.set_xlabel("DMD center (spectral radius)"); ax2.set_ylabel("Cosine similarity")
#     ax2.set_title("DMD-fused similarity vs center"); ax2.grid(True, alpha=0.3, axis="y")

#     plt.tight_layout()
#     plot_path = os.path.join(save_root, f"{tag}_allcenters_plot.png")
#     plt.savefig(plot_path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"   [Plot] → {plot_path}")




#     # ── 横向汇总表格 ─────────────────────────────────────────────────────────
#     _cw   = 9
#     _cols = ["No_DMD"] + ["c=" + str(c) for c in centers]
#     _vals = [nodmd_mean] + [results_by_center[c]["mean"] for c in centers]
#     _hdr  = "  " + f"{'Model':<30}" + "".join(f"{h:>{_cw}}" for h in _cols)
#     _row  = "  " + f"{model_tag:<30}" + "".join(f"{v:>{_cw}.4f}" for v in _vals)
#     _sep  = "-" * len(_hdr)
#     print(f"\nLanguage Ablation  ratio={keep_ratio}")
#     print(_sep); print(_hdr); print(_sep); print(_row); print(_sep + "\n")

#     return {
#         "model": model_name,
#         "keep_ratio": keep_ratio,
#         "layer_sims": layer_sims.tolist(),
#         "results_by_center": {str(c): r["mean"] for c, r in results_by_center.items()},
#     }


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name",  type=str,   default="bert-base-uncased")
#     parser.add_argument("--csv_path",    type=str,   default="data/language_data/EN/lppEN_word_information.csv")
#     parser.add_argument("--save_root",   type=str,   default="results/ablation_lang")
#     parser.add_argument("--keep_ratio",  type=float, default=0.5,  help="保留多少比例的词，0.5=保留50%%")
    
#     parser.add_argument("--device",      type=str,   default="mps")
#     args = parser.parse_args()

#     res = run_ablation(
#         args.model_name, args.csv_path, args.save_root,
#         args.keep_ratio, args.device
#     )

#     print(f"\nSUMMARY: keep={res['keep_ratio']} "
#           "")



"""
Language Context Truncation Ablation
Full sentence vs randomly kept words (scatter) → CLS embedding cosine similarity
"""

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from core.dmd import fuse_layers_single_soft_dmd


def load_lang_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()
    return tokenizer, model


def random_word_keep(words, keep_ratio, mask_token="[MASK]"):
    """随机保留 keep_ratio 比例的词，其余替换为 mask_token，保持句子长度不变。"""
    keep = max(1, int(len(words) * keep_ratio))
    keep_idx = set(np.random.choice(len(words), size=keep, replace=False))
    return [words[i] if i in keep_idx else mask_token for i in range(len(words))]


def build_sentence_windows(words, win_size=50, step=25):
    sentences = []
    for i in range(0, max(1, len(words) - win_size + 1), step):
        sentences.append(words[i:i + win_size])
    if not sentences:
        sentences = [words]
    return sentences


def get_cls_embedding(sentence, tokenizer, model, device):
    """单句 → 每层 CLS token embedding, shape (L, d)"""
    inputs = tokenizer(
        sentence, return_tensors="pt",
        truncation=True, max_length=128
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states
    return np.stack(
        [h[:, 0, :].squeeze(0).cpu().numpy() for h in hidden_states],
        axis=0
    ).astype(np.float32)


def get_embeddings_full_vs_masked(words, tokenizer, model, device, keep_ratio=0.5):
    """
    完整句子 vs 随机保留 keep_ratio 比例词后的句子。
    返回 X_full, X_mask: (L, N, d)
    """
    sentences = build_sentence_windows(words)
    full_layers = []
    masked_layers = []

    for i, sent_words in enumerate(tqdm(sentences, desc="Encoding")):
        emb_full = get_cls_embedding(" ".join(sent_words), tokenizer, model, device)
        full_layers.append(emb_full)

        cropped_words = random_word_keep(sent_words, keep_ratio)
        emb_mask = get_cls_embedding(" ".join(cropped_words), tokenizer, model, device)
        masked_layers.append(emb_mask)

        if i == 0:
            print(f"   [debug] full:    {' '.join(sent_words[:8])}...")
            print(f"   [debug] cropped: {' '.join(cropped_words[:8])}...")

    # list of (L, d) → stack axis=1 → (L, N, d)
    X_full = np.stack(full_layers, axis=1).astype(np.float32)
    X_mask = np.stack(masked_layers, axis=1).astype(np.float32)
    return X_full, X_mask


def cosine_per_layer(X_full, X_mask):
    """(L, N, d) × 2 → per-layer mean cosine sim, shape (L,)"""
    L, N, d = X_full.shape
    sims = []
    for l in range(L):
        A = X_full[l] / (np.linalg.norm(X_full[l], axis=1, keepdims=True) + 1e-8)
        B = X_mask[l] / (np.linalg.norm(X_mask[l], axis=1, keepdims=True) + 1e-8)
        sims.append((A * B).sum(axis=1).mean())
    return np.array(sims)


def dmd_fuse_samples(X_LNd, k=3, center=1.0, sigma=0.1):
    """(L, N, d) → (N, d) via soft DMD fusion"""
    L, N, d = X_LNd.shape
    fused = np.zeros((N, d), dtype=np.float32)
    for n in range(N):
        fused[n] = fuse_layers_single_soft_dmd(X_LNd[:, n, :], r=k, center=center, sigma=sigma)
    return fused


def cosine_sim(A, B):
    """(N, d) × 2 → (N,)"""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (A_norm * B_norm).sum(axis=1)


def plot_results(layer_sims, dmd_sim_mean, dmd_sim_std, model_name, keep_ratio, dmd_center, save_path):
    L = len(layer_sims)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Language Context Ablation — {model_name}\n"
        f"(keep_ratio={keep_ratio}, dmd_center={dmd_center})",
        fontsize=13, fontweight='bold', y=1.01
    )
    ax = axes[0]
    ax.plot(range(L), layer_sims, 'o-', color='#ea580c', lw=2, ms=5)
    ax.axhline(y=layer_sims.mean(), color='gray', ls='--', lw=1.2,
               label=f'mean={layer_sims.mean():.3f}')
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cosine similarity (full vs cropped)")
    ax.set_title("Per-layer similarity")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(['DMD-fused'], [dmd_sim_mean], yerr=[dmd_sim_std],
            color='#0284c7', alpha=0.85, capsize=8, width=0.4,
            error_kw={'lw': 2})
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title(
        f"After DMD fusion (center={dmd_center})\n"
        f"mean={dmd_sim_mean:.3f} ± {dmd_sim_std:.3f}"
    )
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] → {save_path}")


def run_ablation(model_name, csv_path, save_root, keep_ratio=0.5, device="mps"):
    os.makedirs(save_root, exist_ok=True)
    model_tag = model_name.split("/")[-1]

    print("[1/5] Loading words ...")
    df = pd.read_csv(csv_path).sort_values(["section", "onset"])
    sec = sorted(df["section"].unique())[0]
    words = df[df["section"] == sec]["word"].dropna().astype(str).tolist()
    words = [w for w in words if w.strip() and w != "nan"]
    print(f"   → {len(words)} words")

    print("[2/5] Loading model ...")
    tokenizer, model = load_lang_model(model_name, device)

    print("[3/5] Extracting embeddings ...")
    X_full, X_mask = get_embeddings_full_vs_masked(words, tokenizer, model, device, keep_ratio)
    print(f"   shape: {X_full.shape}  (L, N, d)")

    print("[4/5] Per-layer cosine similarity ...")
    layer_sims = cosine_per_layer(X_full, X_mask)
    for l, s in enumerate(layer_sims):
        print(f"   Layer {l:02d}: {s:.4f}")

    print("[5/5] No-DMD baseline + DMD fusion across all centers ...")

    # ── 无DMD基线：所有层均值池化 ────────────────────────────────────────────
    X_full_mean  = X_full.mean(axis=0)        # (N, d)
    X_other_mean = X_mask.mean(axis=0) # (N, d)
    sim_nodmd = cosine_sim(X_full_mean, X_other_mean)
    nodmd_mean = float(sim_nodmd.mean())
    nodmd_std  = float(sim_nodmd.std())
    print(f"   no_dmd (mean pool): {nodmd_mean:.4f} ± {nodmd_std:.4f}")

    centers = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    results_by_center = {}

    for c in centers:
        fused_full = dmd_fuse_samples(X_full, center=c)
        fused_mask = dmd_fuse_samples(X_mask, center=c)
        per_sample_sim = cosine_sim(fused_full, fused_mask)
        sim_mean = float(per_sample_sim.mean())
        sim_std  = float(per_sample_sim.std())
        results_by_center[c] = {"mean": sim_mean, "std": sim_std}
        print(f"   center={c:.1f}: {sim_mean:.4f} ± {sim_std:.4f}")

    tag = f"{model_tag}_keep{keep_ratio}"
    npz_path = os.path.join(save_root, f"{tag}.npz")
    save_dict = {"layer_sims": layer_sims}
    for c, r in results_by_center.items():
        save_dict[f"sim_mean_c{c}"] = np.array([r["mean"]])
        save_dict[f"sim_std_c{c}"]  = np.array([r["std"]])
    np.savez(npz_path, **save_dict)
    print(f"   Saved → {npz_path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"Language Context Ablation — {model_tag}\n(keep_ratio={keep_ratio})",
                 fontsize=13, fontweight="bold", y=1.01)

    ax = axes[0]
    ax.plot(range(len(layer_sims)), layer_sims, "o-", color="#ea580c", lw=2, ms=5)
    ax.axhline(y=layer_sims.mean(), color="gray", ls="--", lw=1.2,
               label=f"mean={layer_sims.mean():.3f}")
    ax.set_xlabel("Layer index"); ax.set_ylabel("Cosine similarity (full vs masked)")
    ax.set_title("Per-layer similarity"); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    c_vals = list(results_by_center.keys())
    means  = [results_by_center[c]["mean"] for c in c_vals]
    stds   = [results_by_center[c]["std"]  for c in c_vals]
    colors = ["#ef4444" if m < 0 else "#3b82f6" for m in means]
    ax2.bar([str(c) for c in c_vals], means, yerr=stds, color=colors,
            alpha=0.8, capsize=6, error_kw={"lw": 1.5})
    ax2.axhline(y=0, color="black", lw=0.8)
    ax2.set_xlabel("DMD center (spectral radius)"); ax2.set_ylabel("Cosine similarity")
    ax2.set_title("DMD-fused similarity vs center"); ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(save_root, f"{tag}_allcenters_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [Plot] → {plot_path}")




    # ── 横向汇总表格 ─────────────────────────────────────────────────────────
    _cw   = 9
    _cols = ["No_DMD"] + ["c=" + str(c) for c in centers]
    _vals = [nodmd_mean] + [results_by_center[c]["mean"] for c in centers]
    _hdr  = "  " + f"{'Model':<30}" + "".join(f"{h:>{_cw}}" for h in _cols)
    _row  = "  " + f"{model_tag:<30}" + "".join(f"{v:>{_cw}.4f}" for v in _vals)
    _sep  = "-" * len(_hdr)
    print(f"\nLanguage Ablation  ratio={keep_ratio}")
    print(_sep); print(_hdr); print(_sep); print(_row); print(_sep + "\n")

    return {
        "model": model_name,
        "keep_ratio": keep_ratio,
        "layer_sims": layer_sims.tolist(),
        "results_by_center": {str(c): r["mean"] for c, r in results_by_center.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str,   default="bert-base-uncased")
    parser.add_argument("--csv_path",    type=str,   default="data/language_data/EN/lppEN_word_information.csv")
    parser.add_argument("--save_root",   type=str,   default="results/ablation_lang")
    parser.add_argument("--keep_ratio",  type=float, default=0.5,  help="保留多少比例的词，0.5=保留50%%")
    
    parser.add_argument("--device",      type=str,   default="mps")
    args = parser.parse_args()

    res = run_ablation(
        args.model_name, args.csv_path, args.save_root,
        args.keep_ratio, args.device
    )

    print(f"\nSUMMARY: keep={res['keep_ratio']} "
          "")