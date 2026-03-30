import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from core.encoder.language_encoder import get_text_embeddings


def generate_language_embeddings(
    csv_path="data/language_data/EN/lppEN_word_information.csv",
    save_root="filterData/lang/design_matrix",
    model_name="bert-base-uncased",
    tr=2.0,
    device=None,
    batch_size=16,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    model_tag = model_name.split("/")[-1]
    save_dir = os.path.join(save_root, model_tag)
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True
    ).to(device)
    model.eval()

    # 读取 CSV 数据
    df = pd.read_csv(csv_path)
    required_cols = {"onset", "offset", "word", "section"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{required_cols - set(df.columns)}")

    df = df.sort_values(["section", "onset"]).reset_index(drop=True)
    sections = sorted(df["section"].unique())


    for sec in sections:
        sub_df = df[df["section"] == sec].reset_index(drop=True)
        words = sub_df["word"].astype(str).tolist()
        n_words = len(words)

        # 获取所有层 embedding
        X_layers = get_text_embeddings(words,device=device,model=model,tokenizer=tokenizer)
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
        print(f"Saved (FP16): {save_path}")

    print(f"\nAll Done! Saved in: {save_dir}")

if __name__ == "__main__":
    lang_models = [
        "albert-base-v2",
        "albert-large-v2",

        "bert-base-cased",
        "bert-base-multilingual-cased",
        "bert-base-uncased",
        "bert-large-cased",
        "bert-large-uncased",

        "microsoft/deberta-base",
        "microsoft/deberta-large",

        "distilbert-base-uncased",

        "google/electra-base-discriminator",
        "google/electra-large-discriminator",

        "roberta-base",       
        "roberta-large",       

        "xlm-roberta-base",    
        "xlm-roberta-large",   
    ]
    for model_name in lang_models:
        generate_language_embeddings(
            model_name=model_name,
            tr=2.0,
            device="mps",
            batch_size=16
        )
