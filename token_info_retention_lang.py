"""
Per-Token Information Retention Across Dynamical Modes (Language)
=================================================================
对每个模型的每个句子：
  1. 提取所有层的 per-token hidden states: (L, N_tokens, D)
  2. 对每个 token 位置独立做 fuse_layers_single_soft_dmd (各 center)
  3. cos(fused_token, last_layer_token) = 信息保留度
  4. 模型内 Friedman test + 模态级 Friedman omnibus + Wilcoxon post-hoc

用法：
  CUDA_VISIBLE_DEVICES=1 python token_info_retention_lang.py --device cuda
  CUDA_VISIBLE_DEVICES=1 python token_info_retention_lang.py --device cuda --max_sentences 30
"""

import argparse
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import gc
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy.stats import friedmanchisquare
from transformers import AutoTokenizer, AutoModel, T5EncoderModel

from core.dmd import fuse_layers_single_soft_dmd


# ═══════════════════════════════════════════════════════════════
#  模型加载
# ═══════════════════════════════════════════════════════════════

def load_lang_model(model_name, device):
    model_lower = model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "t5" in model_lower:
        model = T5EncoderModel.from_pretrained(model_name, output_hidden_states=True)
        model.config.model_type = "t5_encoder"
    else:
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    model = model.to(device).eval()
    return tokenizer, model


# ═══════════════════════════════════════════════════════════════
#  Per-token hidden states 提取
# ═══════════════════════════════════════════════════════════════

def extract_per_token_states(tokenizer, model, sentence, device, max_length=128):
    """单个句子 → (L, N_tokens, D)"""
    inputs = tokenizer(
        sentence, return_tensors="pt",
        truncation=True, max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states
    attn_mask = inputs["attention_mask"][0]
    valid_len = int(attn_mask.sum().item())

    token_states = [h[0, :valid_len, :].cpu().numpy() for h in hidden_states]
    return np.stack(token_states, axis=0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  工具
# ═══════════════════════════════════════════════════════════════

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def format_p(p):
    if p < 0.0001:   return "< 0.0001"
    elif p < 0.001:  return "< 0.001"
    elif p < 0.01:   return "< 0.01"
    elif p < 0.05:   return "< 0.05"
    else:            return f"{p:.4f}"


# ═══════════════════════════════════════════════════════════════
#  核心：处理一个句子的所有 token
# ═══════════════════════════════════════════════════════════════

def process_tokens(token_states, centers, dmd_k=3, sigma=0.1):
    L, N_tokens, D = token_states.shape
    last_layer = token_states[-1]

    sims = {c: [] for c in centers}
    for t in range(N_tokens):
        trajectory = token_states[:, t, :]
        for c in centers:
            try:
                fused = fuse_layers_single_soft_dmd(trajectory, r=dmd_k, center=c, sigma=sigma)
                sims[c].append(cosine_sim(fused, last_layer[t]))
            except Exception:
                sims[c].append(0.0)
    return sims


# ═══════════════════════════════════════════════════════════════
#  构建句子
# ═══════════════════════════════════════════════════════════════

def build_sentences_from_csv(csv_path, win_size=50, step=25):
    df = pd.read_csv(csv_path).sort_values(["section", "onset"])
    all_sentences = []
    for sec in sorted(df["section"].unique()):
        words = df[df["section"] == sec]["word"].dropna().astype(str).tolist()
        words = [w for w in words if w.strip() and w != "nan"]
        for i in range(0, max(1, len(words) - win_size + 1), step):
            all_sentences.append(" ".join(words[i:i + win_size]))
    return all_sentences


# ═══════════════════════════════════════════════════════════════
#  单个模型
# ═══════════════════════════════════════════════════════════════

def run_one_model(model_name, sentences, centers, device="cuda",
                  sigma=0.1, dmd_k=3):

    tokenizer, model = load_lang_model(model_name, device)
    tag = model_name.split("/")[-1]

    all_sims = {c: [] for c in centers}

    for sentence in tqdm(sentences, desc=f"  {tag}"):
        try:
            token_states = extract_per_token_states(tokenizer, model, sentence, device)
        except Exception:
            continue
        if token_states.shape[1] < 3:
            continue
        sims = process_tokens(token_states, centers, dmd_k, sigma)
        for c in centers:
            all_sims[c].extend(sims[c])

    n_total = len(all_sims[centers[0]])
    if n_total < 100:
        print(f"    ⚠️ {tag}: 数据不足 ({n_total})")
        del tokenizer, model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

    # Friedman test
    center_means = {c: float(np.mean(all_sims[c])) for c in centers}
    best_center = max(center_means, key=center_means.get)

    max_n = 50000
    idx = np.random.choice(n_total, size=min(n_total, max_n), replace=False) if n_total > max_n else np.arange(n_total)
    samples = [np.array(all_sims[c])[idx] for c in centers]
    try:
        chi2, p_friedman = friedmanchisquare(*samples)
    except Exception:
        chi2, p_friedman = 0.0, 1.0

    def _sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
    cm_str = " | ".join([f"c={c}:{center_means[c]:.4f}" for c in centers])
    print(f"    {tag:35s} | n={n_total:>6} | {cm_str} | best=c={best_center} | "
          f"χ²={chi2:.1f} p={format_p(p_friedman)}{_sig(p_friedman)}")

    del tokenizer, model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_total": n_total,
        "center_means": center_means,
        "best_center": best_center,
        "friedman_chi2": chi2,
        "friedman_p": p_friedman,
        "sims": {c: np.array(all_sims[c]) for c in centers},
    }


# ═══════════════════════════════════════════════════════════════
#  模态级汇总
# ═══════════════════════════════════════════════════════════════

def aggregate_results(model_results, group_name, centers):
    n = len(model_results)
    n_sig = sum(1 for r in model_results if r["friedman_p"] < 0.05)

    center_vectors = {c: np.array([r["center_means"][c] for r in model_results]) for c in centers}
    center_summary = {c: {"mean": float(np.mean(center_vectors[c])),
                          "std": float(np.std(center_vectors[c], ddof=1)) if n > 1 else 0.0}
                      for c in centers}

    if n >= 3:
        try:
            chi2, p = friedmanchisquare(*[center_vectors[c] for c in centers])
        except Exception:
            chi2, p = 0.0, 1.0
    else:
        chi2, p = 0.0, 1.0

    return {
        "group": group_name,
        "n_models": n,
        "n_significant": n_sig,
        "center_summary": center_summary,
        "friedman_chi2": chi2,
        "friedman_p": p,
        "model_results": model_results,
    }


# ═══════════════════════════════════════════════════════════════
#  表格
# ═══════════════════════════════════════════════════════════════

def print_table1(group_summaries, centers):
    print("\n" + "=" * 150)
    print("Table 1: Per-Token Information Retention — cos(DMD_fused, last_layer) by Center")
    print("  Higher = DMD mode captures more of the final representation at each token position")
    print("=" * 150)

    for gs in group_summaries:
        print(f"\n── {gs['group']} ({gs['n_models']} models, {gs['n_significant']}/{gs['n_models']} significant) ──")
        c_hdrs = " | ".join([f"c={c:>4}" for c in centers])
        header = f"  {'Model':<35} | {'N':>8} | {c_hdrs} | {'Best':>5} | {'χ²':>8} | {'p':>10}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        for r in gs["model_results"]:
            cm = r["center_means"]
            c_vals = " | ".join([f"{cm[c]:>6.4f}" for c in centers])
            name = r["model"].split("/")[-1]
            def _s(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
            print(f"  {name:<35} | {r['n_total']:>8} | {c_vals} | c={r['best_center']:<3} | "
                  f"{r['friedman_chi2']:>8.1f} | {format_p(r['friedman_p']):>10} {_s(r['friedman_p'])}")

    print("=" * 150)


def print_table2(group_summaries, centers):
    from scipy.stats import wilcoxon as wilcoxon_test

    print("\n" + "=" * 115)
    print("Table 2: Cross-Model Universality")
    print("  Friedman omnibus test, followed by Wilcoxon signed-rank post-hoc (two-sided).")
    print("=" * 115)

    def _s(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

    for gs in group_summaries:
        n = gs["n_models"]
        print(f"\n  {gs['group']}  (n = {n} models)")
        print(f"  Friedman χ²({len(centers)-1}) = {gs['friedman_chi2']:.2f},  p = {format_p(gs['friedman_p'])} {_s(gs['friedman_p'])}")
        print()

        header = f"  {'Test':<8} | {'#Models':>7} | {'Mean Δ':>8} | {'Cohen d':>8} | {'W-stat':>8} | {'p-value':>12} | {'Sig':>5}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        vals = {c: np.array([r["center_means"][c] for r in gs["model_results"]]) for c in centers}

        pairs = [
            ("S > T", centers[2], centers[0]),
            ("S > M", centers[2], centers[1]),
            ("M > T", centers[1], centers[0]),
        ]

        for label, c_a, c_b in pairs:
            diff = vals[c_a] - vals[c_b]
            mean_d = float(np.mean(diff))
            std_d = float(np.std(diff, ddof=1)) if n > 1 else 1.0
            cohens_d = mean_d / std_d if std_d > 0 else float('inf')
            if n >= 2 and not np.all(diff == 0):
                w, p = wilcoxon_test(diff, alternative='two-sided')
            else:
                w, p = 0.0, 1.0
            d_str = f"{cohens_d:.2f}" if not np.isinf(cohens_d) else "inf"
            print(f"  {label:<8} | {n:>7} | {mean_d:>+8.4f} | {d_str:>8} | {w:>8.1f} | {format_p(p):>12} | {_s(p):>5}")

        print(f"  {'-' * (len(header) - 2)}")

    print("=" * 115)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

LANGUAGE_MODELS = [
    "albert-base-v2",
    "albert-large-v2",
    "albert-xlarge-v2",
    "nreimers/MiniLM-L6-H384-uncased",
    "sentence-transformers/all-mpnet-base-v2",
    "bert-base-cased",
    "bert-base-multilingual-cased",
    "bert-base-uncased",
    "bert-large-cased",
    "bert-large-uncased",
    "camembert-base",
    "YituTech/conv-bert-base",
    "YituTech/conv-bert-medium-small",
    "facebook/data2vec-text-base",
    "microsoft/deberta-base",
    "microsoft/deberta-large",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased",
    "distilroberta-base",
    "google/electra-base-discriminator",
    "google/electra-large-discriminator",
    "google/electra-small-discriminator",
    "nghuyong/ernie-2.0-base-en",
    "nghuyong/ernie-2.0-large-en",
    "kssteven/ibert-roberta-base",
    "nreimers/MiniLM-L6-H384-uncased",
    "microsoft/mpnet-base",
    "google/rembert",
    "roberta-base",
    "roberta-large",
    "squeezebert/squeezebert-uncased",
    "google-t5/t5-small",
    "xlm-roberta-base",
    "xlm-roberta-large",
    "xlnet-base-cased",
    "xlnet-large-cased",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",         type=str,   default="cuda")
    parser.add_argument("--csv_path",       type=str,   default="data/lang_data/lppEN_word_information.csv")
    parser.add_argument("--sigma",          type=float, default=0.1)
    parser.add_argument("--dmd_k",          type=int,   default=3)
    parser.add_argument("--max_sentences",  type=int,   default=200)
    parser.add_argument("--win_size",       type=int,   default=50)
    parser.add_argument("--step",           type=int,   default=25)
    parser.add_argument("--save_root",      type=str,   default="results/token_retention_lang")
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    centers = [0.0, 0.5, 1.0]

    sentences = build_sentences_from_csv(args.csv_path, args.win_size, args.step)
    sentences = sentences[:args.max_sentences]
    print(f"Sentences: {len(sentences)} (win={args.win_size}, step={args.step})")
    print(f"Centers = {centers}, sigma = {args.sigma}, dmd_k = {args.dmd_k}")

    all_results = []
    for model_name in LANGUAGE_MODELS:
        print(f"\n  [{model_name}]")
        try:
            r = run_one_model(model_name, sentences, centers,
                              device=args.device, sigma=args.sigma, dmd_k=args.dmd_k)
            if r:
                all_results.append(r)
        except Exception as e:
            print(f"    ⚠️ {model_name}: {e}")

    if not all_results:
        print("⚠️ 没有结果")
        return

    gs = aggregate_results(all_results, "Language", centers)
    print_table1([gs], centers)
    print_table2([gs], centers)

    save_data = {}
    for r in all_results:
        mkey = r["model"].replace("/", "_")
        for c in centers:
            save_data[f"{mkey}_c{c}"] = r["sims"][c]

    npz_path = os.path.join(args.save_root, "token_retention_results.npz")
    np.savez(npz_path, **save_data)
    print(f"\n保存: {npz_path}")


if __name__ == "__main__":
    main()