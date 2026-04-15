import torch
import numpy as np
import os
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
import pandas as pd

#CUDA_VISIBLE_DEVICES=2 python new_extract/lang.py
def load_lang_model(model_name="bert-base-uncased", device="cuda"):
    print(f"加载语言模型: {model_name}")
    model_lower = model_name.lower()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT2/XLNet 等没有 pad_token，用 eos_token 代替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "t5" in model_lower:
        model = T5EncoderModel.from_pretrained(model_name, output_hidden_states=True)
        model.config.model_type = "t5_encoder"
    else:
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    model = model.to(device)
    model.eval()
    return tokenizer, model


def get_text_embeddings(words, tokenizer, model, device, batch_size=16):
    """提取每个词的 token 均值 embedding（排除 padding），返回 (n_layers, n_words, dim)"""
    all_layers = []
    model.eval()

    for i in range(0, len(words), batch_size):
        batch_words = words[i:i + batch_size]
        inputs = tokenizer(
            batch_words,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

        # 用 attention_mask 排除 padding，对有效 token 取均值
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, seq, 1)
        mean_layers = [
            ((h * mask).sum(dim=1) / mask.sum(dim=1)).cpu().numpy()
            for h in hidden_states
        ]

        if not all_layers:
            all_layers = [x for x in mean_layers]
        else:
            for li in range(len(mean_layers)):
                all_layers[li] = np.concatenate(
                    [all_layers[li], mean_layers[li]], axis=0
                )

    return np.stack(all_layers, axis=0)


def generate_language_embeddings(
    model_name="bert-base-uncased",
    csv_path="data/lang_data/lppEN_word_information.csv",
    save_root="filterData/lang/design_matrix",
    tr=2.0,
    device="cuda",
    batch_size=16,
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

        X_layers = get_text_embeddings(
            words, tokenizer, model,
            device=device, batch_size=batch_size
        )
        n_layers, n_words, feat_dim = X_layers.shape

        max_time = sub_df["offset"].max()
        n_tr = int(np.ceil(max_time / tr))
        sub_df["tr_idx"] = (sub_df["onset"] / tr).round().astype(int)

        X_TR = np.zeros((n_layers, n_tr, feat_dim), dtype=np.float32)
        for li in range(n_layers):
            for si, row in sub_df.iterrows():
                ti = int(row["tr_idx"])
                if 0 <= ti < n_tr:
                    X_TR[li, ti, :] += X_layers[li, si, :]

        X_TR = X_TR.astype(np.float16)

        save_path = os.path.join(save_dir, f"lppEN_section{sec}_bold_embedding.npy")
        np.save(save_path, X_TR)

    del tokenizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nAll Done! Saved in: {save_dir}")


if __name__ == "__main__":
    lang_models = [
        # ── BERT 系列 ────────────────────────────────
        # "bert-base-uncased",
        # "bert-base-cased",
        # "bert-large-uncased",
        # "bert-large-cased",
        # "bert-base-multilingual-cased",

        # # ── RoBERTa 系列 ─────────────────────────────
        # "roberta-base",
        # "roberta-large",

        # # ── ALBERT 系列 ──────────────────────────────
        # "albert-base-v2",
        # "albert-large-v2",
        # "albert-xlarge-v2",

        # # ── DeBERTa 系列 ─────────────────────────────
        # "microsoft/deberta-base",
        # "microsoft/deberta-large",


        # # ── DistilBERT 系列 ──────────────────────────
        # "distilbert-base-uncased",
        # "distilbert-base-multilingual-cased",

        # # ── ELECTRA 系列 ─────────────────────────────
        # "google/electra-small-discriminator",
        # "google/electra-base-discriminator",
        # "google/electra-large-discriminator",

        # # ── XLM-RoBERTa 系列 ─────────────────────────
        # "xlm-roberta-base",
        # "xlm-roberta-large",

        # # ── XLNet 系列 ───────────────────────────────
        # "xlnet-base-cased",
        # "xlnet-large-cased",

        # # ── T5 encoder 系列 ──────────────────────────
        # "google-t5/t5-small",

        # # ── ERNIE 系列 ───────────────────────────────
        # "nghuyong/ernie-2.0-base-en",
        # "nghuyong/ernie-2.0-large-en",

        # # ── Funnel Transformer 系列 ──────────────────
        # "microsoft/mpnet-base",
        # "sentence-transformers/all-MiniLM-L6-v2",
        # "sentence-transformers/all-mpnet-base-v2",

        # # ── SqueezeBERT ──────────────────────────────
        # "squeezebert/squeezebert-uncased",

        # # ── ConvBERT 系列 ────────────────────────────
        # "YituTech/conv-bert-base",
        # "YituTech/conv-bert-medium-small",


        # # ── I-BERT（量化友好）────────────────────────
        # "kssteven/ibert-roberta-base",

        # # ── Data2Vec-Text 系列 ───────────────────────
        # "facebook/data2vec-text-base",

        # # ── CamemBERT（法语，但架构通用）─────────────
        # "camembert-base",

        # # ── RemBERT ──────────────────────────────────
        # "google/rembert",

        # # ── DistilRoBERTa ────────────────────────────
        # "distilroberta-base",
        "nreimers/MiniLM-L6-H384-uncased",
    ]

    for model_name in lang_models:
        print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
        try:
            generate_language_embeddings(
                model_name=model_name,
                device="cuda",
                batch_size=16
            )
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()