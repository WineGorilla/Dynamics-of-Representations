"""
统一语言 Embedding 提取脚本（纯文本数据集版）
==============================================
支持模型 (36个):
  BERT, ALBERT, RoBERTa, DeBERTa, ELECTRA, XLNet,
  DistilBERT, ConvBERT, SqueezeBERT, MiniLM, MPNet,
  CamemBERT, XLM-RoBERTa, Data2Vec-Text, ERNIE,
  iBERT, RemBERT, T5-encoder, etc.

处理方式:
  词级上下文编码 (context_window 滑窗) + 按固定词数分箱
  输出 shape = (n_layers, n_bins, feat_dim)

输出结构:
  embeddings/language/{model_tag}/{text_file_stem}.npy

用法:
CUDA_VISIBLE_DEVICES=2 python extractnew/extract_language_embeddings.py \
    --data_root /data/mi2-interns/ruiyu/Dynamics-of-Representations/ptb_texts \
    --save_root embeddings \
    --device cuda \
    --context_window 32 \
    --words_per_bin 5
"""

import sys
import os
import gc
import argparse
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel, T5EncoderModel


# ====================================================================
#  文本文件收集
# ====================================================================
TEXT_EXTS = {".txt", ".text"}


def collect_text_paths(data_root):
    """递归收集所有文本文件路径"""
    paths = []
    for root, dirs, files in os.walk(data_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in TEXT_EXTS:
                paths.append(os.path.join(root, f))
    paths.sort()
    print(f"共找到 {len(paths)} 个文本文件 (根目录: {data_root})")
    return paths


def read_words_from_file(file_path):
    """读取文本文件，按空白分词，返回词列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    words = text.split()
    return words


# ====================================================================
#  模型加载
# ====================================================================
def load_lang_model(model_name, device="cuda"):
    print(f"加载语言模型: {model_name}")
    model_lower = model_name.lower()

    # BPE-based 模型需要 add_prefix_space
    needs_prefix_space = any(k in model_lower for k in ["gpt2", "roberta", "bart"])

    tokenizer_kwargs = {"use_fast": True}
    if needs_prefix_space:
        tokenizer_kwargs["add_prefix_space"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "t5" in model_lower:
        model = T5EncoderModel.from_pretrained(model_name, output_hidden_states=True)
        model.config.model_type = "t5_encoder"
    else:
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model = model.to(device).eval()
    return tokenizer, model


# ====================================================================
#  词级上下文编码 (与原脚本逻辑一致)
# ====================================================================
def get_contextual_embeddings(
    words, tokenizer, model, device,
    context_window=32,
    max_length=512,
):
    """
    为每个词提取上下文化 embedding。
    对词 w_i, 用 words[i-context_window : i+1] 作为输入,
    取 w_i 对应 subword tokens 的 hidden state 均值。

    返回: (n_layers, n_words, dim)
    """
    all_layers = None

    for i in tqdm(range(len(words)), desc="  词级编码", ncols=80, leave=False):
        start = max(0, i - context_window)
        context_words = [str(w) for w in words[start:i + 1]]
        target_idx_in_context = len(context_words) - 1

        enc = tokenizer(
            context_words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(device)

        word_ids = enc.word_ids(batch_index=0)
        target_token_positions = [
            t_idx for t_idx, w_id in enumerate(word_ids)
            if w_id == target_idx_in_context
        ]

        # 若因 truncation 被截掉，回退到只用该词本身编码
        if len(target_token_positions) == 0:
            enc = tokenizer(
                [str(words[i])],
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            ).to(device)
            word_ids = enc.word_ids(batch_index=0)
            target_token_positions = [
                t_idx for t_idx, w_id in enumerate(word_ids) if w_id == 0
            ]

        with torch.no_grad():
            outputs = model(**enc)
            hidden_states = outputs.hidden_states

        pos_tensor = torch.tensor(target_token_positions, device=device)
        layer_vecs = []
        for h in hidden_states:
            vec = h[0, pos_tensor, :].mean(dim=0)
            layer_vecs.append(vec.cpu().numpy())
        layer_vecs = np.stack(layer_vecs, axis=0)  # (n_layers, dim)

        if all_layers is None:
            n_layers, feat_dim = layer_vecs.shape
            all_layers = np.zeros((n_layers, len(words), feat_dim), dtype=np.float32)

        all_layers[:, i, :] = layer_vecs

    return all_layers  # (n_layers, n_words, dim)


# ====================================================================
#  按固定词数分箱
# ====================================================================
def bin_by_words(X_layers, words_per_bin=5):
    """
    将词级 embedding 按固定词数分箱取均值。
    输入:  (n_layers, n_words, feat_dim)
    输出:  (n_layers, n_bins, feat_dim)
    """
    n_layers, n_words, feat_dim = X_layers.shape
    n_bins = int(np.ceil(n_words / words_per_bin))

    X_binned = np.zeros((n_layers, n_bins, feat_dim), dtype=np.float32)

    for b in range(n_bins):
        start = b * words_per_bin
        end = min(start + words_per_bin, n_words)
        X_binned[:, b, :] = X_layers[:, start:end, :].mean(axis=1)

    return X_binned


# ====================================================================
#  模型名称映射 (短名称 → HuggingFace ID)  共36个
# ====================================================================
LANG_MODELS = {
    # ── MiniLM / MPNet (4) ──
    "MiniLM-L6-H384-uncased":      "nreimers/MiniLM-L6-H384-uncased",
    "all-MiniLM-L6-v2":            "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2":           "sentence-transformers/all-mpnet-base-v2",
    "mpnet-base":                  "microsoft/mpnet-base",

    # ── ALBERT (3) ──
    "albert-base-v2":              "albert-base-v2",
    "albert-large-v2":             "albert-large-v2",
    "albert-xlarge-v2":            "albert-xlarge-v2",

    # ── BERT (5) ──
    "bert-base-cased":             "bert-base-cased",
    "bert-base-multilingual-cased": "bert-base-multilingual-cased",
    "bert-base-uncased":           "bert-base-uncased",
    "bert-large-cased":            "bert-large-cased",
    "bert-large-uncased":          "bert-large-uncased",

    # ── CamemBERT (1) ──
    "camembert-base":              "camembert-base",

    # ── ConvBERT (2) ──
    "conv-bert-base":              "YituTech/conv-bert-base",
    "conv-bert-medium-small":      "YituTech/conv-bert-medium-small",

    # ── Data2Vec-Text (1) ──
    "data2vec-text-base":          "facebook/data2vec-text-base",

    # ── DeBERTa (2) ──
    "deberta-base":                "microsoft/deberta-base",
    "deberta-large":               "microsoft/deberta-large",

    # ── DistilBERT (2) ──
    "distilbert-base-multilingual-cased": "distilbert-base-multilingual-cased",
    "distilbert-base-uncased":     "distilbert-base-uncased",

    # ── DistilRoBERTa (1) ──
    "distilroberta-base":          "distilroberta-base",

    # ── ELECTRA (3) ──
    "electra-base-discriminator":  "google/electra-base-discriminator",
    "electra-large-discriminator": "google/electra-large-discriminator",
    "electra-small-discriminator": "google/electra-small-discriminator",

    # ── ERNIE (2) ──
    "ernie-2.0-base-en":           "nghuyong/ernie-2.0-base-en",
    "ernie-2.0-large-en":          "nghuyong/ernie-2.0-large-en",

    # ── iBERT (1) ──
    "ibert-roberta-base":          "kssteven/ibert-roberta-base",

    # ── RemBERT (1) ──
    "rembert":                     "google/rembert",

    # ── RoBERTa (2) ──
    "roberta-base":                "roberta-base",
    "roberta-large":               "roberta-large",

    # ── SqueezeBERT (1) ──
    "squeezebert-uncased":         "squeezebert/squeezebert-uncased",

    # ── T5 (1) ──
    "t5-small":                    "google-t5/t5-small",

    # ── XLM-RoBERTa (2) ──
    "xlm-roberta-base":            "xlm-roberta-base",
    "xlm-roberta-large":           "xlm-roberta-large",

    # ── XLNet (2) ──
    "xlnet-base-cased":            "xlnet-base-cased",
    "xlnet-large-cased":           "xlnet-large-cased",
}


def resolve_model_name(name):
    """短名称 → 完整 HuggingFace ID"""
    return LANG_MODELS.get(name, name)


# ====================================================================
#  主流程
# ====================================================================
def run_extraction(
    model_name,
    text_paths,
    save_root,
    device="cuda",
    context_window=32,
    words_per_bin=5,
):
    full_name = resolve_model_name(model_name)
    model_tag = full_name.split("/")[-1]

    model_save_dir = os.path.join(save_root, model_tag)
    os.makedirs(model_save_dir, exist_ok=True)

    tokenizer, model = load_lang_model(full_name, device)

    for text_path in tqdm(text_paths, desc=f"  {model_tag}", ncols=80, leave=False):
        fname = os.path.splitext(os.path.basename(text_path))[0]
        save_path = os.path.join(model_save_dir, f"{fname}.npy")

        if os.path.exists(save_path):
            continue

        try:
            words = read_words_from_file(text_path)
            if len(words) == 0:
                print(f"  跳过空文件: {text_path}")
                continue

            # 词级上下文编码
            X_layers = get_contextual_embeddings(
                words, tokenizer, model,
                device=device,
                context_window=context_window,
            )

            # 按固定词数分箱
            X_binned = bin_by_words(X_layers, words_per_bin=words_per_bin)

            # 保存 fp32（与原脚本一致，DMD 对精度敏感）
            np.save(save_path, X_binned.astype(np.float32))

        except Exception as e:
            print(f"  ❌ Failed {os.path.basename(text_path)}: {e}")
            continue

    del tokenizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ====================================================================
#  CLI
# ====================================================================
ALL_MODEL_NAMES = list(LANG_MODELS.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一语言 Embedding 提取 (36个模型)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="文本数据集根目录（.txt 文件，支持子文件夹）")
    parser.add_argument("--save_root", type=str, default="embeddings",
                        help="输出根目录，会在下面创建 language/{model_tag}/ 子目录")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--context_window", type=int, default=32,
                        help="上下文窗口大小（词数），默认 32")
    parser.add_argument("--words_per_bin", type=int, default=5,
                        help="每个 bin 的词数，默认 5")
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"要跑的模型名称列表，不指定则跑全部。\n可选: {ALL_MODEL_NAMES}")
    return parser.parse_args()


def main():
    args = parse_args()

    # 收集文本文件
    text_paths = collect_text_paths(args.data_root)
    if len(text_paths) == 0:
        print("未找到任何文本文件，退出。")
        return

    # 在 save_root 下创建 language 子目录
    lang_root = os.path.join(args.save_root, "language")
    os.makedirs(lang_root, exist_ok=True)

    # 确定要跑的模型
    model_list = args.models if args.models else ALL_MODEL_NAMES

    # 保存文本路径索引
    index_path = os.path.join(lang_root, "text_paths.txt")
    with open(index_path, "w") as f:
        for p in text_paths:
            f.write(p + "\n")
    print(f"文本索引已保存: {index_path}")

    # 逐模型提取
    for model_name in model_list:
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
        try:
            run_extraction(
                model_name=model_name,
                text_paths=text_paths,
                save_root=lang_root,
                device=args.device,
                context_window=args.context_window,
                words_per_bin=args.words_per_bin,
            )
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n全部完成！输出目录: {lang_root}")


if __name__ == "__main__":
    main()