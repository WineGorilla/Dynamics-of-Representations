import torch
import numpy as np
import os
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
import pandas as pd

# CUDA_VISIBLE_DEVICES=2 python new_extract/lang_new.py


def load_lang_model(model_name="bert-base-uncased", device="cuda"):
    print(f"加载语言模型: {model_name}")
    model_lower = model_name.lower()

    # 判断是否为 BPE-based 模型(需要 add_prefix_space)
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

    # GPT-2 等 causal 模型需要显式设 pad_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model = model.to(device)
    model.eval()
    return tokenizer, model


def get_contextual_embeddings(
    words, tokenizer, model, device,
    context_window=32,
    max_length=512,
):
    """
    为每个词提取上下文化 embedding。
    对词 w_i,用 words[i-context_window : i+1] 作为输入,
    取 w_i 对应 subword tokens 的 hidden state 均值。

    返回: (n_layers, n_words, dim)
    """
    all_layers = None  # 延迟初始化,等知道 n_layers/dim

    for i in tqdm(range(len(words)), desc="  词级编码", ncols=80, leave=False):
        start = max(0, i - context_window)
        context_words = [str(w) for w in words[start:i + 1]]
        target_idx_in_context = len(context_words) - 1  # 目标词在上下文里的位置

        # 关键:is_split_into_words=True 让我们能用 word_ids() 定位
        enc = tokenizer(
            context_words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(device)

        word_ids = enc.word_ids(batch_index=0)
        # 找到 target word 的所有 subword token 位置
        target_token_positions = [
            t_idx for t_idx, w_id in enumerate(word_ids)
            if w_id == target_idx_in_context
        ]

        # 若因 truncation 被截掉(上下文很长时末尾通常保留,但稳妥起见做兜底)
        if len(target_token_positions) == 0:
            # 回退:只用该词本身编码
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
            hidden_states = outputs.hidden_states  # tuple of (1, seq, dim)

        # 对每一层,取 target token 位置的 hidden state 均值
        pos_tensor = torch.tensor(target_token_positions, device=device)
        layer_vecs = []
        for h in hidden_states:
            # h: (1, seq, dim)
            vec = h[0, pos_tensor, :].mean(dim=0)  # (dim,)
            layer_vecs.append(vec.cpu().numpy())
        layer_vecs = np.stack(layer_vecs, axis=0)  # (n_layers, dim)

        if all_layers is None:
            n_layers, feat_dim = layer_vecs.shape
            all_layers = np.zeros((n_layers, len(words), feat_dim), dtype=np.float32)

        all_layers[:, i, :] = layer_vecs

    return all_layers  # (n_layers, n_words, dim)


def generate_language_embeddings(
    model_name="bert-base-uncased",
    csv_path="data/lang_data/lppEN_word_information.csv",
    save_root="filterData/lang_new/design_matrix",
    tr=2.0,
    device="cuda",
    context_window=32,
):
    model_tag = model_name.split("/")[-1]
    save_dir = os.path.join(save_root, model_tag)
    os.makedirs(save_dir, exist_ok=True)

    tokenizer, model = load_lang_model(model_name, device)

    df = pd.read_csv(csv_path)
    required_cols = {"onset", "offset", "word", "section"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"缺少列: {required_cols - set(df.columns)}")

    df = df.sort_values(["section", "onset"]).reset_index(drop=True)
    sections = sorted(df["section"].unique())

    for sec in tqdm(sections, desc=f"{model_tag}", ncols=80):
        sub_df = df[df["section"] == sec].reset_index(drop=True)
        words = sub_df["word"].astype(str).tolist()

        # ── 步骤 A:词级上下文编码 ─────────────────────
        X_layers = get_contextual_embeddings(
            words, tokenizer, model,
            device=device,
            context_window=context_window,
        )
        n_layers, n_words, feat_dim = X_layers.shape

        # ── 步骤 B:按 TR 分箱(floor),对同一 TR 内的词取均值 ─
        max_time = sub_df["offset"].max()
        n_tr = int(np.ceil(max_time / tr)) + 1  # +1 留点余量
        tr_idx = np.floor(sub_df["onset"].values / tr).astype(int)

        X_TR = np.zeros((n_layers, n_tr, feat_dim), dtype=np.float32)
        count = np.zeros(n_tr, dtype=np.int32)

        for si in range(n_words):
            ti = tr_idx[si]
            if 0 <= ti < n_tr:
                X_TR[:, ti, :] += X_layers[:, si, :]
                count[ti] += 1

        # 求均值(空 TR 保持为 0)
        nonzero = count > 0
        X_TR[:, nonzero, :] /= count[nonzero][None, :, None]

        # 保存为 fp32(DMD 对精度敏感,不再用 fp16)
        save_path = os.path.join(save_dir, f"lppEN_section{sec}_bold_embedding.npy")
        np.save(save_path, X_TR.astype(np.float32))

    del tokenizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nAll Done! Saved in: {save_dir}")


if __name__ == "__main__":
    lang_models = [
        # "albert-base-v2",
        # "albert-large-v2",
        # "albert-xlarge-v2",
        # "nreimers/MiniLM-L6-H384-uncased",
        # "sentence-transformers/all-mpnet-base-v2",
        # "bert-base-cased",
        # "bert-base-multilingual-cased",
        # "bert-base-uncased",
        # "bert-large-cased",
        # "bert-large-uncased",
        # "camembert-base",
        # "YituTech/conv-bert-base",
        # "YituTech/conv-bert-medium-small",
        # "facebook/data2vec-text-base",
        # "microsoft/deberta-base",
        # "microsoft/deberta-large",
        # "distilbert-base-multilingual-cased",
        # "distilbert-base-uncased",
        # "distilroberta-base",
        # "google/electra-base-discriminator",
        # "google/electra-large-discriminator",
        # "google/electra-small-discriminator",
        # "nghuyong/ernie-2.0-base-en",
        # "nghuyong/ernie-2.0-large-en",
        # "kssteven/ibert-roberta-base",
        # "microsoft/mpnet-base",
        # "google/rembert",
        # "roberta-base",
        # "roberta-large",
        # "squeezebert/squeezebert-uncased",
        # "google-t5/t5-small",
        # "xlm-roberta-base",
        # "xlm-roberta-large",
        # "xlnet-base-cased",
        # "xlnet-large-cased",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]

    for model_name in lang_models:
        print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
        try:
            generate_language_embeddings(
                model_name=model_name,
                device="cuda",
                context_window=32,
            )
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()